# Non-Invasive Anemia + Malnutrition Detection — Project Summary

**Problem statement:** Non-invasive early detection of **anemia** and **malnutrition** in **children (<5 yrs)** and **pregnant women**, using a Raspberry Pi 5 (16GB) + Pi Cam 3 + mic/speaker + electronics kit.

---

## 1. Hardware Available

| Item | Use |
|---|---|
| Raspberry Pi 5 (16GB) | On-device inference (CNN + TFLite), local storage, control hub |
| 512GB SSD | Storage for models, patient data, logs |
| Pi Cam 3 | Pallor imaging, MUAC measurement, edema tracking |
| Mic + Speaker | Voice command layer (offline STT/TTS) |
| Breadboard/LEDs/buttons/load cell etc. | PPG LED setup, edema probe, weight sensor, UI buttons |

---

## 2. What "Non-Invasive" Means (and module audit)

**Definition:** No skin puncture, no blood draw, no internal probing. Camera, pressure, sound, and questions all count as non-invasive.

| Module | Invasive? |
|---|---|
| Conjunctiva / nail / tongue pallor (camera) | ✅ Non-invasive |
| MUAC (camera + reference card) | ✅ Non-invasive |
| Weight (load cell) | ✅ Non-invasive |
| Edema grid-sheet press | ✅ Non-invasive |
| Questionnaire / voice layer | ✅ Non-invasive |
| Optional home BLE finger-prick Hb meter | ⚠️ **Invasive** (minimally) — must be pitched as an *optional bonus fusion feature*, not part of the core non-invasive claim |

---

## 3. How Doctors Actually Diagnose These Conditions Today

### 3a. Anemia — clinical exam signs
- Conjunctival pallor, nail bed pallor, palmar crease pallor, oral/tongue pallor
- Tachycardia, tachypnea/breathlessness on exertion
- Koilonychia (spoon nails), glossitis, angular stomatitis

### 3b. Anemia — lab tests (gold standard)
| Test | Tells you |
|---|---|
| Hemoglobin (Hb) | Main anemia marker |
| Hematocrit (Hct) | % blood volume that's RBCs |
| MCV | Cell size → type of anemia (micro/macrocytic) |
| RBC count | Number of red cells |
| RDW | Cell-size variation — rises **before** Hb drops (earliest signal) |
| Ferritin, iron, TIBC | Confirms iron-deficiency; ferritin drops **before** Hb |
| Peripheral smear | Cell shape under microscope |
| Reticulocyte count | Bone-marrow compensation |

### 3c. Point-of-care comparators
- **HemoCue** — handheld finger-prick device, Hb in ~1 min
- **Masimo Pronto (pulse CO-oximeter)** — non-invasive multi-wavelength finger clip, similar principle to PPG idea, expensive & moderate accuracy

### 3d. Malnutrition — clinical tools
| Tool | Use |
|---|---|
| MUAC tape | WHO field/screening standard, esp. children <5 |
| Weight-for-height / BMI | Compared to WHO growth charts / BMI cutoffs |
| Height-for-age | Stunting (chronic) indicator |
| Serum albumin / prealbumin | Protein status (prealbumin = faster early marker) |
| Micronutrient panels | Vitamin D, B12, zinc, iron |

**Key insight:** Clinical exam + MUAC = *screening*, not diagnosis — it decides who gets a lab test. The project's device automates this screening layer, not the lab/CBC step.

---

## 4. Diagnosis Flow — Malnutrition (Children)

| Stage | Tool | Data | Compared to | Deterministic / Probabilistic |
|---|---|---|---|---|
| 1. MUAC | Color-coded tape | cm value | WHO cutoffs: <11.5=SAM(red), 11.5–12.5=MAM(yellow), ≥12.5=normal | **Deterministic** |
| 2. Bilateral pitting edema | Thumb press, 3 sec | Pits / doesn't pit | Direct clinical sign | **Deterministic** — overrides everything, auto-SAM if positive |
| 3. Weight + Height → Z-scores | Scale, stadiometer | WFH / HFA / WFA Z-scores | **WHO Child Growth Standards** ([who.int/tools/child-growth-standards](https://www.who.int/tools/child-growth-standards)) | **Statistical/probabilistic** (percentile-based) |
| 4. Visual signs | Inspection | Hair/skin/wasting checklist | Clinical judgment | Qualitative, corroborates |
| 5. Appetite test (if SAM) | Standardized food sample | Pass/fail | Body-weight ratio | Treatment routing |
| 6. Labs (only if flagged) | Blood glucose, Hb, electrolytes, prealbumin | — | Complication check | Reactive, not first-line |

**Order of operations:** MUAC → edema check → Z-scores → visual signs → appetite test → labs.

### Device mapping
| Clinical stage | Implementation |
|---|---|
| MUAC | Pi Cam + reference card → pixel-to-cm calibration |
| Edema | Grid-sheet + camera deformation tracking |
| Z-scores | Load cell (weight) + camera/manual height + WHO LMS lookup tables |
| Visual signs | CNN classifier (hair/skin cues) |
| Appetite test / labs | Out of scope — "refer to clinician" |

---

## 5. Diagnosis Flow — Malnutrition (Pregnant Women)

| Stage | Tool | Data | Compared to | Det./Prob. |
|---|---|---|---|---|
| 1. MUAC | Tape | cm | Single cutoff: **<23 cm = at risk** | Deterministic |
| 2. Pre-pregnancy BMI | Height + baseline weight | BMI | <18.5 underweight / 18.5–24.9 normal / ≥25 overweight | Deterministic, one-time baseline |
| 3. Gestational weight gain trend | Scale each visit | Weight over time | **IOM gestational weight-gain charts** ([NAP.edu, IOM 2009](https://www.nationalacademies.org/read/12584/chapter/2); summary via [ACOG](https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2013/01/weight-gain-during-pregnancy)) | **Trend/slope — needs 2+ visits** |
| 4. Fundal height | Tape on abdomen | cm ≈ weeks gestation | Expected-for-GA rule of thumb | Rough deterministic, noisy |
| 5. Clinical exam | Inspection | Pallor, edema, hair/skin | Edema is a **confounder** here (pregnancy edema can be normal; pitting + high BP → preeclampsia, not malnutrition) | Qualitative |
| 6. Lab panel | Hb, ferritin, albumin, glucose, folate/B12, Ca/Vit D/Zn | — | Routine/proactive **every visit**, unlike children (reactive only) | Mixed |

### Key structural difference from children
- Children → single-visit diagnosis possible (Z-score is a snapshot).
- Pregnant women → **needs persistent per-user history** (login + local storage) because the real signal is the *trend* across visits, not one reading.

---

## 6. Diagnosis Flow — Anemia (Children)

| Stage | Tool | Data | Compared to | Det./Prob. |
|---|---|---|---|---|
| 1. History/symptoms | Verbal | Fatigue, pica, poor appetite, infections | — | Qualitative, raises suspicion |
| 2. Physical exam (pallor) | Inspection | Conjunctiva/nail/palm/tongue | — | Subjective, poor sensitivity for mild cases |
| 3. CBC | Blood draw | Hb, Hct, RBC, MCV, RDW | **WHO age-band Hb cutoffs** ([WHO 2024 Hb-cutoff guideline](https://www.who.int/publications/i/item/9789240088542)) | **Deterministic threshold** |
| 4. MCV classification | — | Cell size | <80fL microcytic (iron def.) / 80–100 normal / >100 macrocytic (B12/folate) | Deterministic branch |
| 5. Follow-up (if microcytic) | Ferritin, iron/TIBC, smear, electrophoresis | — | Confirms cause | Reactive |

**Early-detection insight:** RDW rises and ferritin drops *before* Hb crosses the cutoff — a "pre-anemic" stage that a single Hb snapshot misses.

---

## 7. Diagnosis Flow — Anemia (Pregnant Women)

| Stage | Detail |
|---|---|
| 1. Screening trigger | **Mandatory & routine** every trimester (not symptom-triggered like children) |
| 2. Physical exam | Same pallor checks, but less reliable due to pregnancy physiology |
| 3. Hb cutoffs | Trimester-specific: **<11 g/dL (T1/T3), <10.5 g/dL (T2)** — due to hemodilution (plasma volume rises faster than RBC mass) ([WHO Hb-cutoff guideline, 2024](https://www.who.int/publications/i/item/9789240088542)) |
| 4. MCV classification | Same as children; iron deficiency far more common (fetal demand) |
| 5. Follow-up tests | Ferritin, folate/B12 (routine/proactive, not just reactive), electrophoresis (thalassemia/sickle-cell screen) |
| 6. Early-detection insight | Watch **trend across trimesters** — a drop bigger than the expected physiological dip is the real red flag |

### Device mapping (both anemia flows)
| Clinical stage | Implementation |
|---|---|
| Pallor exam | Pi Cam + CV/CNN — core anemia module |
| Hb lab value | Cannot be replicated non-invasively — camera gives a **risk estimate**, not an actual Hb number |
| Age/trimester thresholds | Software lookup table |
| RDW/ferritin | Out of scope — flag "needs lab test" |
| Trend tracking | Needs persistent patient history (pregnancy module only) |

---

## 8. Vision (Pi Cam) Checks — Consolidated

| Check | Condition | Region | Method | Output | Type |
|---|---|---|---|---|---|
| Conjunctival pallor | Anemia | Inner eyelid | Color-space (RGB/HSV) analysis, CNN regression, based on approaches like **HemaApp** ([UW paper, ACM UbiComp 2016](https://dl.acm.org/doi/10.1145/2971648.2971653); [UW News summary](https://www.washington.edu/news/2016/09/07/hemaapp-screens-for-anemia-blood-conditions-without-needle-sticks/)) | Hb risk score | Probabilistic |
| Nail bed pallor | Anemia | Fingernail | Press/release color + refill time | Refill score | Probabilistic |
| Palmar crease pallor | Anemia | Palm | Color contrast | Secondary score | Probabilistic |
| Tongue/oral mucosa | Anemia | Tongue | Color + texture (glossitis) | Pallor + glossitis flag | Probabilistic |
| MUAC | Malnutrition | Mid-upper arm | Reference-object pixel calibration + contour tracing | cm → WHO cutoff | **Deterministic** |
| Visible wasting | Malnutrition | Body/limbs | CNN classifier | Present/absent | Probabilistic |
| Hair texture/color | Malnutrition | Scalp | CNN classifier | Chronic flag | Probabilistic |
| Skin lesions/flaking | Malnutrition | Skin | CNN classifier | Micronutrient flag | Probabilistic |
| Facial wasting | Malnutrition | Cheeks | Landmark contour measurement | Corroborating signal | Semi-deterministic |
| Edema (foot/shin) | Malnutrition | Foot/pretibial | Grid-sheet distortion tracking | Pitting grade 1+–4+ | **Deterministic** |
| Fundal height (stretch) | Malnutrition (pregnant) | Abdomen | Camera/marker distance | Fetal growth proxy | Unreliable via camera — manual entry recommended |

**MVP priority order:** MUAC (pure geometry) → conjunctival pallor CNN → edema grid tracking → hair/skin/wasting classifiers (stretch).

---

## 9. Edema ("Foot-Dent") Module — Formalized

This maps to real clinical **pitting-edema grading (1+ to 4+)** based on depth + rebound time.

| Approach | Setup | Notes |
|---|---|---|
| **A — Camera optical deformation** | Grid/dot pattern on flexible silicone sheet over shin/foot; Pi Cam captures baseline (flat), then post-press recovery frame-by-frame | No extra hardware; essentially structured-light depth estimation |
| **B — Force sensor array** | FSR strips + IR/ultrasonic distance sensor | Simpler math, needs added ~$5–10 sensors |

| Grade | Depth | Recovery time |
|---|---|---|
| 1+ | Barely visible | Immediate |
| 2+ | Slight | <15 sec |
| 3+ | Moderate | 15–30 sec |
| 4+ | Deep | >30 sec, visible >30 min |

**Synthetic data trick:** Use MATLAB/Python to generate synthetic contour/depth maps per grade (varying skin tone, noise, lighting) to bootstrap training data before real data collection — a legitimate "synthetic data generation" pitch point.

**Pregnancy caveat:** edema here is a confounder — pitting + high BP needs preeclampsia workup, not a pure malnutrition signal (no BP cuff currently in kit; scope out or flag "refer if edema + high BP").

---

## 10. Questionnaires (Risk-Weighting Layer, Not a Diagnosis)

General rule: metadata questions (age, trimester, BMI) route to the right reference chart; scoring questions add to a **0–1 weighted risk score**.

```
Risk Score = Σ (weight_i × answer_value_i)     [weights sum to 1.0]
```

### Anemia — Children
| Question | Weight |
|---|---|
| Pica (ice/dirt/chalk) | 0.25 |
| Fatigue/weakness/irritability | 0.20 |
| Recent diarrhea/parasitic infection | 0.20 |
| Low iron-rich food intake | 0.15 |
| Premature/low birth weight | 0.10 |
| Slow growth/delayed milestones | 0.10 |

Cutoffs: <0.3 low · 0.3–0.6 moderate · >0.6 high (recommend lab test)

### Anemia — Pregnant Women
| Question | Weight |
|---|---|
| Previous pregnancy anemia | 0.20 |
| Fatigue/dizziness/breathlessness | 0.20 |
| Not taking iron/folic acid | 0.20 |
| Heavy periods pre-pregnancy | 0.15 |
| Vegetarian/vegan, low iron | 0.10 |
| Twin/multiple pregnancy | 0.10 |
| GI absorption condition | 0.05 |

Cutoffs: <0.25 low · 0.25–0.55 moderate · >0.55 high

### Malnutrition — Children
| Question | Weight |
|---|---|
| Dietary diversity (WHO Minimum Dietary Diversity-style, 24h food groups) | 0.25 |
| Meals/day below age minimum | 0.15 |
| Illness/diarrhea (2 wks) | 0.15 |
| Household food insecurity | 0.15 |
| Not exclusively breastfed (0–6mo) | 0.10 |
| Milestones not met | 0.10 |
| Vaccination not up to date | 0.05 |
| Low birth weight | 0.05 |

Cutoffs: <0.3 low · 0.3–0.6 moderate · >0.6 high

### Malnutrition — Pregnant Women
| Question | Weight |
|---|---|
| Dietary diversity (24h) | 0.25 |
| Not taking prenatal supplements | 0.20 |
| Severe nausea/vomiting | 0.20 |
| High parity (3+ births) | 0.15 |
| Physically demanding occupation | 0.10 |
| Chronic condition (confounder flag) | 0.10 |

Cutoffs: <0.3 low · 0.3–0.6 moderate · >0.6 high

### Combining questionnaire + sensor data
```
Final Risk = (α × Questionnaire_Score) + (β × Sensor/CV_Confidence_Score)
Suggested: α = 0.3, β = 0.7   (hard measurement dominates; questionnaire nudges borderline cases)
```
These weights are initial clinically-informed estimates — meant to be refit once real paired (questionnaire + lab outcome) data exists.

---

## 11. Full Question Bank (reference — not all used in MVP demo)

| Category | Sample extra questions |
|---|---|
| Anemia – children | Diet type, family history of thalassemia/sickle cell (routes to electrophoresis, not scored) |
| Anemia – pregnant | Interpregnancy interval, menstrual history |
| Malnutrition – children | Birth weight, vaccination status, developmental milestones |
| Malnutrition – pregnant | Pre-pregnancy weight/height, parity, occupation, twin gestation |
| Cross-cutting (all 4) | Socioeconomic status, rural/urban, prior diagnosis history, current medications/supplements |

**Build note:** MVP should use only the 4–5 highest-signal questions per category; list the rest as "planned expansion" in the pitch.

---

## 12. System Architecture

### Clinical Mode
- One Pi 5 unit per department, role-based login (RFID/PIN)
- Pediatrics login → only child records; OB-GYN login → only pregnancy records
- Local SQLite/PostgreSQL, synced to hospital server if available; every access logged (audit trail)
- Flow: scan → local inference → tagged to patient ID → doctor reviews/overrides

### Home Mode
- Simplified consumer device / CHW kit
- **Optional** Bluetooth Hb-meter pairing → fuses real Hb reading with camera pallor estimate (fusion weight β≈0.9 toward real data when present)
- Simple "normal / consult a doctor" output, no raw clinical numbers
- Fully offline-capable; optional sync to clinic record

### Voice Control Layer (fully offline)
| Component | Tool |
|---|---|
| Wake word + STT | Vosk or Whisper.cpp (tiny/base) |
| TTS | Piper |
| Fixed commands | "Start MUAC scan," "Start pallor scan," "Read last result," "Save and next patient," "Repeat," "Cancel" |

Fixed-intent design (not open NLU) keeps it reliable for low-connectivity/low-literacy use.

---

## 13. Consolidated Data/Model Table — All 4 Combinations

| | Anemia – Children | Anemia – Pregnant | Malnutrition – Children | Malnutrition – Pregnant |
|---|---|---|---|---|
| **Data needed** | Conjunctiva/nail image, age | + trimester, prior Hb history | MUAC, weight, height, age, hair/skin image | MUAC, weight trend, pre-preg BMI, fundal height, edema grade |
| **Core sensor** | Pi Cam | Pi Cam | Pi Cam + load cell | Pi Cam + load cell + edema sheet |
| **Model type** | CNN regression (probabilistic) | Same + trend model | Geometry (MUAC, deterministic) + Z-score lookup + optional CNN | Same + weight-gain trend regression + edema CV |
| **Ground truth** | WHO age-band Hb cutoffs | WHO trimester Hb cutoffs | WHO Z-score charts + MUAC cutoffs | IOM weight-gain charts + MUAC <23cm |
| **Single-session viable?** | Yes | Better with history | Yes | **No — needs history** |
| **Persistent storage** | Optional | Required | Optional | Required |

---

## 14. End-to-End Case Walkthroughs

### Case 1: Child Malnutrition Screening (CHW, village health camp)
1. Voice: "Start child screening" → device wakes into child mode
2. Patient ID created/pulled from SQLite
3. Metadata Q&A (age, sex) → selects WHO chart
4. Risk questionnaire → Questionnaire_Score = 0.58 (moderate-high)
5. MUAC scan (reference card + contour) → 11.8 cm → **yellow (MAM)**
6. Weight (8.2 kg) + height (74 cm) → WFH Z = −2.3 (moderate wasting), HFA Z = −1.1 (normal)
7. Edema grid check → no pitting
8. Hair/skin CNN scan → mild dryness, low-confidence flag
9. Fusion: `Final Risk = 0.7×(sensor severity) + 0.3×(questionnaire)` → **Moderate Acute Malnutrition, escalating**
10. Output spoken + logged under patient ID; CHW says "Save and next patient"

### Case 2: Pregnant Woman — Anemia Screening (2nd trimester, returning patient)
1. Voice: "Start pregnancy screening" → RFID/PIN login (OB-GYN role)
2. Loads Session 1 data for trend comparison
3. Metadata: 24 weeks → selects 10.5 g/dL trimester-2 cutoff
4. Risk questionnaire → Questionnaire_Score = 0.51
5. Pallor scan (conjunctiva + nail) → CV estimate: Hb 9.5–10.2 g/dL, confidence 0.72
6. Optional home BLE Hb meter → real reading 9.8 g/dL → **overrides CV estimate** (β≈0.9)
7. Trend check: Session 1 Hb 11.2 → Session 2 Hb 9.8 → drop **larger than expected hemodilution** → red flag
8. Classification: Hb 9.8 < 10.5 cutoff → anemia confirmed, escalate priority
9. If doctor enters manual CBC MCV → routes to "likely iron deficiency, recommend ferritin test"
10. Output spoken + logged, trend graph on dashboard; "Save and schedule follow-up"

**What both cases show:** same shared architecture (voice → questionnaire → vision/sensor → fusion scoring → role-gated storage) across all 4 disease/demographic combinations — **one system, 4 configurable diagnostic modes**, not 4 separate tools.

---

## 15. Research Gaps Targeted

1. No integrated single-device pipeline covering **both** anemia and malnutrition non-invasively
2. Lighting/skin-tone bias in most pallor models (limited training diversity)
3. No real-time **on-device** inference (most apps upload to cloud)
4. Little CV-based MUAC automation, almost none paired with anemia detection in one session
5. No audio/speech layer for fatigue/breathlessness cues combined with visual data (novel multimodal angle)
6. Weak field validation specifically for pregnant women (most datasets skew to children/general population)

**Strongest hackathon angle:** multimodal fusion (image + PPG + MUAC + voice + questionnaire → single risk score), running fully local on Pi 5.

---

## 16. Honest Scoping Notes for the Pitch

- **Deterministic, trustworthy outright:** MUAC, weight/height Z-scores, edema depth/recovery — direct physical measurements, no ML uncertainty.
- **Probabilistic, present as a risk flag:** Pallor-based Hb estimation — a trained model, not a diagnostic replacement; pair with optional home BLE fusion to boost confidence.
- **Out of scope, correctly excluded:** CBC, ferritin, RDW, appetite test, full lab panel — device flags "refer to clinician" instead.
- The **only invasive component** is the optional home BLE finger-prick Hb meter — must be framed as a bonus fusion layer, never as part of the core non-invasive claim.

---

## 17. References & Resources Cited in This Chat

| Resource | Link |
|---|---|
| WHO Child Growth Standards (Z-score charts) | https://www.who.int/tools/child-growth-standards |
| WHO MUAC cutoffs for SAM/MAM (2013 guideline, as cited in literature) | https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7144967/ |
| WHO Guideline on Haemoglobin Cutoffs to Define Anaemia (2024) | https://www.who.int/publications/i/item/9789240088542 |
| WHO/NLIS Anaemia data & indicators | https://www.who.int/data/nutrition/nlis/info/anaemia |
| IOM Gestational Weight Gain Guidelines (2009), National Academies | https://www.nationalacademies.org/read/12584/chapter/2 |
| ACOG summary of IOM weight-gain guidelines | https://www.acog.org/clinical/clinical-guidance/committee-opinion/articles/2013/01/weight-gain-during-pregnancy |
| HemaApp — noninvasive Hb via smartphone camera (UW, ACM UbiComp 2016, Best Paper) | https://dl.acm.org/doi/10.1145/2971648.2971653 |
| HemaApp — UW News summary | https://www.washington.edu/news/2016/09/07/hemaapp-screens-for-anemia-blood-conditions-without-needle-sticks/ |
| HemaApp v2 extension paper (PDF) | https://ubicomplab.cs.washington.edu/pdfs/hemaapp_v2.pdf |

*(Apps referenced conversationally — Eyenaemia, Anemocheck India, Masimo Pronto, HemoCue — are commercial/clinical products mentioned by name in the original discussion; no official links were verified for these and they should be independently checked before citing in a pitch deck.)*

---

## 18. ASCII Flow Diagram — Full System

```
                              ┌─────────────────────────┐
                              │   VOICE LAYER (offline)  │
                              │  Vosk/Whisper.cpp (STT)  │
                              │  Piper (TTS)              │
                              └────────────┬─────────────┘
                                           │ "Start child/pregnancy screening"
                                           ▼
                         ┌──────────────────────────────────┐
                         │  SESSION START + PATIENT ID       │
                         │  (RFID/PIN login → role-based)    │
                         │  Load prior history if returning  │
                         └───────────────┬────────────────────┘
                                         │
                     ┌───────────────────┼────────────────────┐
                     ▼                                        ▼
        ┌─────────────────────────┐             ┌─────────────────────────┐
        │  QUESTIONNAIRE (voice)  │             │  METADATA CAPTURE       │
        │  weighted Yes/No items  │             │  age/sex/trimester/BMI  │
        │  → Questionnaire_Score  │             │  → selects ref. chart   │
        │      (0–1)              │             └─────────────────────────┘
        └────────────┬─────────────┘
                     │
                     ▼
        ┌─────────────────────────────────────────────────────────┐
        │                  SENSOR / VISION CAPTURE                 │
        │                                                          │
        │  ANEMIA:                    MALNUTRITION:                │
        │  • Conjunctiva/nail/tongue  • MUAC (camera + ref card)   │
        │    pallor → CNN (Pi Cam)    • Weight (load cell) +       │
        │  • Optional PPG (LED+cam)     Height → WHO Z-scores      │
        │  • Optional BLE Hb meter    • Edema grid-sheet (Pi Cam)  │
        │    (⚠ invasive, optional)   • Hair/skin CNN checks       │
        └────────────────────────────┬───────────────────────────┘
                                     │
                                     ▼
                    ┌───────────────────────────────────┐
                    │        FUSION SCORING ENGINE       │
                    │ Final Risk = α·Questionnaire        │
                    │            + β·Sensor/CV confidence │
                    │        (α=0.3, β=0.7 default)       │
                    │ + trend check (pregnancy: multi-    │
                    │   visit history comparison)         │
                    └───────────────────┬─────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │     CLASSIFICATION + OUTPUT         │
                    │  Normal / Moderate / Severe / Refer │
                    │  Spoken (Piper) + on-screen          │
                    └───────────────────┬─────────────────┘
                                        │
                                        ▼
                    ┌───────────────────────────────────┐
                    │   ROLE-GATED STORAGE (SQLite/PG)    │
                    │   Patient ID + timestamp + CHW/     │
                    │   doctor ID, audit trail            │
                    │   (optional sync to hospital server)│
                    └───────────────────┬─────────────────┘
                                        │
                                        ▼
                         "Save and next patient" /
                         "Save and schedule follow-up"
```

---
