# Stage 1 Design — Voice Questionnaire Engine (STT → Branching Logic → TTS)

## 1. Why the current scaffold can't do this yet

`questionnaire.default.json` right now is a **flat list** — 3 questions, no
branching, no population split. `App.jsx` renders all of them at once with no
skip logic. STT/TTS routers (`stt.py`, `tts.py`) just return a status string —
there's no actual audio pipeline. This stage is about turning that flat list
into a **decision-tree questionnaire** driven by voice, using the real
weighting scheme already defined in `References.md` section 10.

## 2. Core idea: a question graph, not a question list

Model the questionnaire as a **directed graph** stored in JSON config
(same "config-first" pattern the scaffold already uses). Each node is a
question with:

- `id`
- `applies_if` — a condition referencing earlier answers (population gate)
- `weight` — risk weight (only for scored questions, per the tables in
  References.md §10)
- `next` — optional explicit branch override; otherwise engine falls through
  to the next eligible node in order

The engine walks this graph one node at a time. At each step it:
1. Evaluates `applies_if` against answers collected so far.
2. If false → skip node, move to next.
3. If true → TTS speaks the question → STT captures the answer → answer is
   validated/normalized → stored → loop.

This is a **rule-based state machine**, not an LLM/NLU system — matches the
project's own "fixed-intent design... reliable for low-connectivity/low-
literacy use" principle (References.md §12).

## 3. Population routing — the actual branching logic

First 2 questions are always asked and always gate everything else:

```
Q0: age_years        (number)
Q1: is_pregnant       (yes/no — only asked if age_years >= 12, else skip → assumed "no")
```

From `age_years` + `is_pregnant`, derive a single `population` bucket before
any other question is asked:

```
if age_years < 5                      → "child_under5"     (malnutrition + anemia, child track)
elif age_years < 12                   → "child_5_12"        (anemia track mainly; MUAC child-style)
elif age_years >= 12 and is_pregnant  → "pregnant_woman"
else                                   → "adult_nonpregnant" (out of MVP scope — short exit message)
```

This single `population` value is the master gate. Every subsequent question
declares which population(s) it applies to:

```json
{
  "id": "previous_pregnancy_anemia",
  "applies_to": ["pregnant_woman"],
  "category": "anemia",
  "weight": 0.20
}
```

If `population` isn't in `applies_to`, the engine skips the question
silently — this is exactly your "age is 21 → skip child-only checks" example.

## 4. Two parallel scored tracks per population

Per References.md §10, each population has **two independent risk scores**
running at once — anemia and malnutrition — each its own weighted sum:

```
anemia_score        = Σ (weight_i × answer_i)   for anemia-category questions
malnutrition_score   = Σ (weight_i × answer_i)   for malnutrition-category questions
```

Cutoffs (from the tables, per population) turn each into low/moderate/high.
The questionnaire node list is just every row from the 4 tables in §10,
tagged with `population` + `category` + `weight`. Nothing needs to be
invented — it's a direct JSON transcription of tables already in your theory
doc.

## 5. Conditional sub-branches within a track (beyond population gating)

A few questions are conditional on a *previous answer*, not just population,
e.g.:
- "Twin/multiple pregnancy" only matters once we know they're pregnant (already
  gated by population) — no further branching needed.
- "Not exclusively breastfed (0–6mo)" for children only makes sense if
  `age_months <= 6` — so `child_under5` questions should ask `age_months`
  early and gate this one further with `applies_if: age_months <= 6`.
- "GI absorption condition" (anemia, pregnant) could optionally trigger a
  one-off follow-up like "diagnosed condition name" — free-text, not scored,
  just logged for the clinician review screen.

So `applies_if` needs to support two kinds of conditions:
- **Population membership** (`applies_to: [...]`) — the coarse gate.
- **Answer-value condition** (`applies_if: {question: "age_months", op: "<=", value: 6}`) — fine-grained gate on a specific prior answer.

Keep this rule engine dead simple — no arbitrary code execution, just a small
set of comparison operators (`==`, `!=`, `<`, `<=`, `>`, `>=`, `in`) evaluated
against the answers dict. That's enough to express every branch mentioned in
References.md.

## 6. End-of-questionnaire output

Once the graph is exhausted, the engine emits a structured result — this is
the object `prediction.py` and later the non-invasive test stage will consume:

```json
{
  "population": "pregnant_woman",
  "answers": { "age_years": 24, "is_pregnant": "yes", "fatigue": "yes", ... },
  "scores": {
    "anemia": { "raw": 0.42, "band": "moderate" },
    "malnutrition": { "raw": 0.18, "band": "low" }
  },
  "next_stage": "non_invasive_tests"
}
```

This becomes the input contract for Stage 2 (the pallor/MUAC/edema camera
models) — each of which is only relevant for certain (population, band)
combinations, same skip logic pattern continues downstream.

## 7. STT/TTS plumbing (kept intentionally dumb for Stage 1)

- **TTS**: speak `question.label` (Piper, offline, per References.md §12).
- **STT**: capture short utterance (Vosk or whisper.cpp tiny/base, per same
  section), map to nearest valid option via simple fuzzy match against
  `option.label`/synonyms list (e.g. "yeah", "haan", "yes" → `yes`). For
  `type: number`, run through a numeral parser (word-to-number for spoken
  digits).
- Fixed-intent commands stay available throughout: "repeat", "cancel",
  "skip" (only where `required: false`).
- `STT_PROVIDER=mock` / `TTS_PROVIDER=mock` (already in `.env.example`) stay
  as the dev-mode fallback — mock STT reads answers from a debug text field,
  mock TTS just renders text in the UI. This lets you build and demo the
  branching logic **before** wiring real audio models — good match for your
  "plug and play, if model doesn't exist say so and skip" philosophy, just
  applied one layer earlier (STT/TTS engines instead of vision models).

## 8. Config file shape (concrete extension of existing schema)

Extend `questionnaire.default.json` from a flat `questions` array into:

```json
{
  "questionnaire_id": "adaptive-screening-v1",
  "entry_questions": ["age_years", "is_pregnant"],
  "population_rule": { ... derivation logic as data, see §3 ... },
  "questions": [
    {
      "id": "fatigue",
      "type": "select",
      "label": "Do you feel unusually tired or weak?",
      "applies_to": ["pregnant_woman", "child_5_12"],
      "category": "anemia",
      "weight": 0.20,
      "options": [
        {"label": "No", "value": "no", "score": 0},
        {"label": "Yes", "value": "yes", "score": 1}
      ]
    }
  ]
}
```

This is backward-compatible with the existing loader (`read_json_object`) —
just a richer schema. `app/flows/questionnaire.py` needs a real engine module
(`app/flows/questionnaire_engine.py`) instead of just serving the raw JSON —
that's the actual Stage-1 build target.
