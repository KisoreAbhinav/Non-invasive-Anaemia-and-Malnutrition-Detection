# Pitting-edema simulation demo

A standalone FastAPI + browser-canvas sandbox that demonstrates the *analogy* of a 3-second pitting-edema press: normal tissue rebounds quickly; edema-mode tissue rebounds slowly and retains a residual dent. It is not imported by, connected to, or deployed with the screening-device project next to it.

## Safety and scope

Every record this application creates is synthetic. The fabric is an educational viscoelastic model, not a validated tissue model, and neither detector is a medical device or a diagnostic method. Clinical pitting-edema interpretation depends on examination context. Edema during pregnancy in particular can be a confounder and requires appropriate clinical assessment.

The deterministic verdict is a transparent demo rule: it marks a pit when post-release depth remains above a configurable fraction of the press peak at a configurable time. The visible vision status captures an unindented synthetic camera frame, then identifies a connected dark concavity in later rendered frames. It is an explanatory visualization only, not evidence of clinical performance.

## Run

```bash
cd "Simulation Demo"
python -m venv .venv
.venv/bin/pip install -e .
.venv/bin/uvicorn app.main:app --reload --port 8090
```

Open `http://127.0.0.1:8090`.

The generated local files are intentionally ignored by version control:

- `data/synthetic_patients.jsonl` — one complete synthetic patient record per line
- `data/classifier.json` — the optional lightweight classifier used by the API experiments

## Design notes

- `app/physics.py` is the one press/recovery parameterized model used by the live API and all generated records. Normal and edema use parameter sets, not separate code paths.
- The browser turns the returned centre-point depth into a Gaussian-deformed connected canvas grid and draws the same returned curve. Its visible vision detector stores an unindented canvas frame, uses frame differencing to isolate a connected dark concavity, and derives the detection box centre and scale from that image region. Clicking the canvas clears the active frame and detector state.
- The synthetic lab-panel directions follow the supplied project field schema: lower Hb/ferritin/albumin and related nutrition markers for the positive label, with deliberately overlapping random intervals; TIBC is skewed higher with the iron-deficiency-style positive profile. The values are intentionally marked illustrative rather than reference ranges.
- The optional API classifier turns five fixed post-release points into tiny synthetic intensity maps, then derives brightness features. It never feeds the classifier’s hidden tissue parameters, mode, depth curve, or deterministic verdict.

## Reference grounding

The host project’s `theory/References.md` supplied the demo’s stated clinical framing: a three-second edema press and depth/rebound-time grading, with 2+ described as under 15 seconds, 3+ as 15–30 seconds, and 4+ as longer. The [WHO IMAI clinician manual](https://iris.who.int/bitstream/handle/10665/77751/9789241548290_Vol2_eng.pdf) describes applying firm pressure for a few seconds and identifying pitting when an indentation persists after release. The WHO’s [2024 haemoglobin-cutoff guideline](https://www.who.int/publications/i/item/9789240088542) is the appropriate source for real anaemia thresholds; thresholds vary with population, physiology, altitude and other factors. The [NIH iron health-professional fact sheet](https://ods.od.nih.gov/factsheets/Iron-HealthProfessional/) supports the directional relationship between iron deficiency, ferritin, Hb/Hct, and TIBC. These sources do not validate this application’s synthetic values or visual detector.

## Test

```bash
cd "Simulation Demo"
.venv/bin/pip install -e . pytest httpx
.venv/bin/pytest
```
