You are working inside an existing FastAPI + React + Docker scaffold for a
non-invasive anemia/malnutrition screening device (Raspberry Pi 5 target).
The repo already has: `backend/app/main.py`, independent flow routers under
`backend/app/flows/` (`questionnaire.py`, `stt.py`, `tts.py`, `prediction.py`,
`runtime.py`), config-first JSON at `backend/config/`, and a React scaffold
at `frontend/src/App.jsx`. Docker Compose runs backend on :8000 and frontend
on :8080. Keep this architecture — do not restructure it.

## Task: implement Stage 1 only — the real voice questionnaire engine

Text-to-speech asks each question out loud, speech-to-text records and
transcribes the spoken answer, and a matching layer maps that transcription
to a valid answer value. The engine decides the _next_ question adaptively
based on everything answered so far. Nothing beyond this — no camera/vision
models, no MUAC, no edema, no prediction models.

---

## 1. STT — Vosk, Indian English, plug-and-play model folder

- Add a `backend/models/vosk/` directory (gitignored, empty at repo init
  except a `.gitkeep`). This is where I will manually drop a downloaded
  **Vosk Indian English model** folder (e.g. `vosk-model-en-in-0.5` or
  similar — whatever folder name the extracted model has).
- On startup, the STT engine scans `backend/models/vosk/` for a subfolder
  that looks like a valid Vosk model (contains the expected Vosk model
  files/structure — `am/`, `graph/`, `conf/` etc.). If found, load it once
  and keep it in memory (Vosk model loads are expensive — do not reload per
  request).
- If no valid model folder is found: STT status endpoint reports
  `"model": "not_found"`, and `/transcribe` returns a clear error
  (`503` with `{"error": "no STT model found in backend/models/vosk — drop a
Vosk model folder there and restart"}`), rather than crashing. This is the
  same plug-and-play "if the model doesn't exist, say so and skip" pattern
  used for the later vision models — apply it here too.
- `STT_PROVIDER` env var: `mock` (existing passthrough) or `vosk` (new, real
  engine). Default stays `mock` for CI/dev without the model present;
  `docker-compose.yml`/`.env.example` should document both.
- Implement actual Vosk inference: accept a raw audio chunk (16kHz mono PCM
  WAV — document this requirement clearly, since Vosk expects it), run it
  through `KaldiRecognizer`, return the transcribed text plus Vosk's
  confidence/partial-result info if available.
- Add `vosk` as a backend dependency (`pyproject.toml` / `uv add vosk`).

## 2. TTS — Piper, auto-setup, slowed rate, Indian-accent voice

- Add an auto-setup step (a small script, e.g.
  `backend/scripts/setup_piper_voice.py`, runnable manually and also called
  from the Dockerfile build) that:
  - Checks if a Piper voice model already exists under
    `backend/models/piper/`.
  - If not, downloads a Piper voice with an **Indian English accent** (e.g.
    an `en_IN` Piper voice from the official Piper voices release — pick the
    best available `en_IN` voice at build time; if none exists, clearly
    document falling back to the closest neutral `en` voice and log a
    warning) — both the `.onnx` model and its `.onnx.json` config file.
  - This keeps with the plug-and-play spirit but for TTS it's fully
    automated (no manual placement needed, unlike Vosk) since Piper voices
    are small and directly downloadable — clearly comment why STT and TTS
    setup differ.
- Implement real Piper inference: load the voice once at startup, synthesize
  speech for a given `text`, return audio (WAV bytes) from
  `POST /api/flows/tts/speak`.
- **Slow the speech rate down** for accessibility/comprehension — use
  Piper's `length_scale` parameter (values >1.0 slow speech down; start
  around `1.15`–`1.3` and make it configurable via a `TTS_LENGTH_SCALE` env
  var, default `1.2`) rather than hardcoding.
- `TTS_PROVIDER` env var: `mock` (existing) or `piper` (new, real engine).
  If `piper` is selected but the voice model wasn't downloaded/found, fall
  back to the same "not found, here's why, here's what to do" error pattern
  as STT — don't crash silently.
- Add `piper-tts` (or the appropriate Piper Python binding/CLI wrapper) as a
  backend dependency.

## 3. Answer matching — fuzzy string match first, sentence-transformers fallback

Raw STT transcriptions of short spoken answers ("yeah", "haan ji", "no not
really", "I've been really tired lately") need to map onto the fixed set of
valid option values in the questionnaire config. Build a two-tier matcher in
a new module `backend/app/flows/answer_matcher.py`:

- **Tier 1 — fuzzy lexical match** (fast, cheap, no model load): use
  `rapidfuzz` to compare the transcription against each option's `label`
  plus a small `synonyms` list you add to the question schema (e.g. for a
  yes/no question: `yes` synonyms = `["yes", "yeah", "yep", "haan",
"haanji", "correct", "affirmative"]`; `no` synonyms = `["no", "nah",
"nahi", "negative", "not really"]`). If the best fuzzy match score is
  above a high-confidence threshold (e.g. `>85`), accept it immediately —
  no need for the heavier model.
- **Tier 2 — semantic match via sentence-transformers** (fallback for when
  fuzzy match is inconclusive, e.g. free-form phrasing that doesn't share
  surface tokens with any synonym): embed the transcription and each
  option's label/synonyms using a small `sentence-transformers` model
  (e.g. `all-MiniLM-L6-v2` — small enough to run on a Pi), compute cosine
  similarity, pick the best match if it clears a second, slightly lower
  confidence threshold (e.g. `>0.6`). Load this model once at startup,
  lazily, only if it's actually needed (don't force-load it if a
  questionnaire session never hits Tier 2) — respect Pi resource
  constraints.
- If **neither tier** produces a confident match, don't guess: return an
  "unclear, please repeat" result so the engine can re-ask the question
  (reuse the existing "repeat" fixed-intent command instead of inventing a
  new flow).
- For `type: number` questions (e.g. age), skip the fuzzy/semantic matcher
  entirely — use a word-to-number parser instead (spoken numerals like
  "twenty four" → `24`), with a fallback to extracting digit tokens directly
  from the transcription if that fails.
- Add `rapidfuzz` and `sentence-transformers` as backend dependencies.
- Unit test this module in isolation
  (`backend/tests/test_answer_matcher.py`) with representative Indian
  English phrasings for each question type in the config — this module
  should not require FastAPI, Vosk, or Piper to be running to test.

## 4. Adaptive branching — decide the next question from answers so far

Rewrite `backend/config/questionnaire.default.json` into a branching schema
(not a flat list):

- `entry_questions`: `age_years`, then `is_pregnant` (only asked when age
  makes it relevant).
- Derive a `population` bucket from those two answers before anything else
  is asked: `child_under5` (age < 5), `child_5_12` (5–12), `pregnant_woman`
  (12+ and pregnant), `adult_nonpregnant` (12+ and not pregnant — out of MVP
  scope, short exit message, no further questions).
- Every other question declares `applies_to: [population, ...]`. The engine
  walks the question list in order and silently skips any question whose
  `applies_to` doesn't include the derived population — e.g. age 21 with
  `is_pregnant: no` never triggers "how many times have you been pregnant."
- Support a finer per-question `applies_if` condition referencing a _prior
  answer value_ (not just population), e.g. `{"question": "age_months",
"op": "<=", "value": 6}` for "not exclusively breastfed" — only relevant
  under 6 months old. Support operators `==`, `!=`, `<`, `<=`, `>`, `>=`,
  `in`. Keep this a tiny, safe comparison evaluator — no arbitrary code
  execution.
- Populate the actual question content (all 4 population×condition
  combinations, weights, cutoff bands) from `theory/References.md` section
  10 verbatim — don't invent new numbers.
- Implement `backend/app/flows/questionnaire_engine.py` as pure,
  framework-agnostic functions: `derive_population(answers)`,
  `next_question(schema, answers, population)`, `score(schema, answers,
population)`. Unit test in `backend/tests/test_questionnaire_engine.py`
  covering population derivation, skip logic, and score/band computation.

## 5. Session API — ties STT/TTS/matching/branching together

In `backend/app/flows/questionnaire.py`, add:

- `POST /api/flows/questionnaire/session` → starts a session (in-memory
  dict keyed by session id — no DB yet), returns session id + first
  question text.
- `POST /api/flows/questionnaire/session/{id}/ask` → runs TTS on the current
  question's `label`, returns the audio (or, in mock mode, just the text).
- `POST /api/flows/questionnaire/session/{id}/answer` → accepts recorded
  audio (or raw text in mock mode), runs it through STT → answer_matcher →
  validated value, stores it, advances via `next_question`, returns either
  the next question or, if done, the final result:
  `{population, answers, scores: {anemia: {raw, band}, malnutrition: {raw,
band}}, next_stage: "non_invasive_tests"}`.
- `GET /api/flows/questionnaire/session/{id}` → current state, for
  resume/debug.
- Wire a `"repeat"` fixed command: if the matcher returns "unclear," the
  session state re-serves the same question on the next `/ask` call instead
  of advancing.

## 6. Frontend — drive the real session, mic + audio playback

Update `frontend/src/App.jsx` (or split into `QuestionnaireFlow.jsx`) to:

- Start a session on mount, show one question at a time.
- Play the TTS audio for the current question (or show text in mock mode).
- Record a short audio clip via the browser mic (basic `MediaRecorder`,
  16kHz mono if possible, or resample before sending) and POST it to
  `/answer`; in mock mode, fall back to a plain text input instead.
- Show the matched value back to the user before advancing (so they can
  correct/retry via "repeat" if the transcription was wrong).
- On completion, render the final population + both scores/bands.
- Keep the UI minimal — this is still scaffold-quality, not final UX.

## 7. Don't touch

- `backend/app/flows/prediction.py` / `prediction-models.default.json`
  (later stage — same plug-and-play "no model found → skip" pattern applies
  there too, but isn't part of this task).
- MUAC, pallor imaging, edema, or any camera/vision model.
- Docker/Compose beyond adding the new env vars documented above
  (`STT_PROVIDER=vosk`, `TTS_PROVIDER=piper`, `TTS_LENGTH_SCALE`), and
  ensuring the Piper voice auto-setup step runs during `docker compose
build`.

## 8. Acceptance check

With a real Vosk Indian English model dropped into `backend/models/vosk/`
and `STT_PROVIDER=vosk`, `TTS_PROVIDER=piper` set:

- `docker compose up --build` succeeds; Piper voice auto-downloads during
  build if not already present.
- Starting a session speaks the first question at the slowed rate in the
  Indian-accent voice.
- Speaking "ten" or "10" for age routes to `child_under5`/`child_5_12` and
  never asks pregnancy-specific questions.
- Speaking "twenty four" then "haan" for pregnant routes to
  `pregnant_woman` and never asks child-only questions.
- An unclear/mumbled answer triggers a re-ask instead of a wrong silent
  match.
- Finishing either flow returns population + both category scores/bands
  matching the cutoffs in `theory/References.md` §10.
- Removing the Vosk model folder and restarting causes `/transcribe` to
  return the documented "model not found" error instead of crashing, and
  `STT_PROVIDER=mock` still works end-to-end with no models present at all.
