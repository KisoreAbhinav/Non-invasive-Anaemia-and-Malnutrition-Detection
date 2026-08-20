# Model-inference merge audit

## Scope and source state

- **Merge base:** local remote-tracking commit `origin/main` at `49a2eb4`. No fetch or pull was performed.
- **Target layout:** the application lives under `project/` on `main`. The feature checkout keeps the same application files at its repository root, so every ported path is rebased under `project/`.
- **Feature source:** the on-disk `NiviCopy/Non-invasive-Anaemia-and-Malnutrition-Detection/` checkout of `feat/model-inference` at `f7a9525`, including its newer, uncommitted trained pallor exports. Existing feature-checkout changes were not edited or cleaned.
- **Out of scope:** `Simulation Demo/` is main-only and unrelated. It must remain unchanged. Training datasets, local virtual environments, caches, logs, and checkpoints are development inputs rather than runtime dependencies and will not be copied.

## File and folder audit

Line-ending-only differences were ignored. Local environments, dependency caches, build output, bytecode, logs, and the thousands of source training images were excluded from this application-level comparison.

### Present only in the feature source

- Inference and tests: `backend/app/flows/vision_inference.py`, `backend/tests/test_vision_inference.py`.
- Model setup: `backend/scripts/setup_vision_models.py`.
- Runtime artifacts and metadata under `backend/models/vision/`: `edema`, `hair_skin`, `pallor`, `pallor_eye`, `pallor_nail`, and `pallor_palm` model pairs, plus timestamped pallor backups.
- Training references and tooling: `training/README.md`, `training/requirements.txt`, `training/train.py`, `training/train_all.py`, `training/train_all_v2.py`, `training/export.py`, six JSON configs, and empty data-folder markers.
- Training source data and checkpoints also exist on disk but are deliberately excluded from the runtime merge. The deployed TorchScript files are self-contained; inference does not read a checkpoint, spreadsheet, or training image.
- Feature-only handoff/design notes: `HANDOFF_REPORT.md` and `docs/superpowers/`. These are source-history notes, not application dependencies.

### Shared files with substantive differences

- Backend: `backend/app/flows/model_registry.py`, `backend/app/flows/screening.py`, `backend/config/visual-cues.default.json`, `backend/pyproject.toml`, `backend/uv.lock`, and `backend/tests/test_screening_flow.py`.
- Frontend: `frontend/src/App.jsx` and `frontend/src/index.css`.
- `.gitignore` adds training output and local model rules.
- Most other apparent differences are CRLF-versus-LF only and must not be copied over main.

### Present only in main

- At repository level, `Simulation Demo/` and main's root project documentation are main-only.
- Within `project/`, no tracked application source file is absent from the feature source. Main's local installation additionally contains downloaded Piper, Vosk, and sentence-transformer assets; these are ignored runtime dependencies and are not replaced.

## Model inventory

All six active `model.pt` files in the feature checkout are CPU-loadable PyTorch **TorchScript** modules. Each accepts a float tensor shaped `[1, 3, 224, 224]`, emits two logits shaped `[1, 2]`, and is postprocessed with softmax. Parameter counts were obtained by loading each TorchScript module and summing its parameters.

| Directory | Architecture / parameters | Input preprocessing | Output | Status |
| --- | --- | --- | --- | --- |
| `pallor_eye` | MobileNetV3-Large, 4,204,594 | RGB image; resize shorter side to 256; center crop 224; standards-based sRGB -> CIE Lab; scale L to `[0,1]`, a/b to `[0,1]`; normalize with mean `[.5,.5,.5]`, std `[.5,.25,.25]` | `normal`, `risk`; primary `risk` | Fine-tuned export; reported validation accuracy 0.626. |
| `pallor_nail` | MobileNetV3-Large, 4,204,594 | Same as `pallor_eye` | `normal`, `risk`; primary `risk` | Fine-tuned export; reported validation accuracy 0.969. |
| `pallor_palm` | MobileNetV3-Large, 4,204,594 | Same as `pallor_eye` | `normal`, `risk`; primary `risk` | Fine-tuned export; reported validation accuracy 0.994. |
| `pallor` | MobileNetV3-Small, 1,519,906 | Direct RGB resize to 224; ImageNet normalization | `normal`, `risk`; primary `risk` | Placeholder: ImageNet weights plus an untrained replacement head. Superseded by the three site models. |
| `edema` | MobileNetV3-Small, 1,519,906 | Direct RGB resize to 224; ImageNet normalization | `normal`, `pitting_edema` | Placeholder: not domain-fine-tuned. |
| `hair_skin` | EfficientNet-B0, 4,010,110 | Direct RGB resize to 224; ImageNet normalization | `normal`, `signs_detected` | Placeholder: not domain-fine-tuned. |

The three pallor directories also contain timestamped small/large backups. Only the canonical `model.pt` and `model.json` pair is loaded. Backup files and training checkpoints are not needed at inference time.

### Loading and invocation

`backend/app/main.py` calls `model_registry.initialize_registry()` at startup. The registry scans immediate directories under `settings.vision_models_path` (default `backend/models/vision`), validates each `model.json`, and lazily calls `torch.jit.load(..., map_location="cpu")` on first use. `vision_inference.predict_image()` retrieves the metadata/model, preprocesses bytes, invokes the model under `torch.inference_mode()`, applies the declared activation, and maps logits to the declared labels.

## Inference path trace

1. `frontend/src/App.jsx` receives the visual test plan. A model-backed test uses browser `getUserMedia`; pallor renders eye, nail-bed, and palm capture slots. File upload is the fallback.
2. Each captured camera frame is center-cropped to a square JPEG in a hidden canvas. Pallor sends a multipart form with keys `pallor_eye`, `pallor_nail`, and `pallor_palm`. Single-image tests send a raw JPEG body.
3. `POST /api/flows/screening/vision/{test_id}?population=...` verifies population and applicability from the plan. Multi-site tests require every configured part.
4. `predict_batch()` invokes the corresponding model for each site. `_image_tensor()` decodes with Pillow, resizes/crops according to model metadata, performs RGB or CIE Lab conversion, scales, and normalizes.
5. Each two-logit output is softmaxed. The API returns named scores, primary risk score, classification, confidence, provenance, and (for pallor) per-image results. Pallor combines the three sites with an unweighted arithmetic mean of each named score.
6. The UI renders the combined and per-site scores. It stores `primary_score` separately from top-class confidence so final sensor fusion consumes anemia risk rather than confidence in whichever class won.

### Assumptions, hardcoded values, and thresholds

- Supported model tensors are hardcoded to batch 1 and three channels; metadata supplies spatial dimensions.
- Pallor metadata declares resize 256 and center crop 224. The resize follows torchvision's shorter-side semantics for square resize values.
- Lab conversion uses D65 constants and explicit channel scaling. PIL's `convert("LAB")` is not equivalent.
- Browser capture JPEG quality is 0.95 and camera preference is 1280x720, environment-facing.
- Pallor aggregation is a simple 1/3 mean; it has no calibrated decision threshold. Classification is `argmax`, effectively a 0.5 binary boundary.
- Model availability is currently inferred solely from a structurally valid artifact pair. Provenance is not considered, which allows placeholders to be advertised and executed.
- The feature UI auto-start effect is tied to `stage` rather than the active `testIndex`, so the camera may not reopen reliably after advancing between tests.

## Root cause

The feature history contained several independent failure modes rather than one model-loading problem.

1. **A hard preprocessing crash.** Revision `fbb10e3` renamed resize variables to `target_w`/`target_h` but still reshaped with undefined `height`/`width` names. Replaying that code produced `NameError: name 'height' is not defined` before the model call.
2. **Incorrect aspect-ratio resize and crop.** The next revision used `min(target_w / width, target_h / height)` for square shorter-side resize. A 640x480 frame therefore became 256x192, and its requested 224 crop began at `(16, -16)`, padding 32 rows instead of matching torchvision's 341x256 resize and valid center crop. This changed the image distribution and could suppress the relevant body site.
3. **Training/serving color-space skew.** The first Lab inference implementation used Pillow's byte-encoded `Image.convert("LAB")`, while the final training pipeline uses a continuous D65 sRGB -> linear RGB -> XYZ -> CIE Lab transform with explicit L/a/b scaling. The two encodings are not interchangeable around neutral a/b values, where the pallor signal is subtle. Feeding the wrong encoding yielded unreliable/near-boundary behavior even when the model loaded.
4. **Wrong artifacts advertised as usable models.** `edema`, `hair_skin`, and the generic `pallor` artifacts are placeholder pretrained backbones with replacement two-class heads, not domain-trained models. The registry only validates file shape/metadata, so the UI advertises them as available and returns arbitrary but plausible-looking probabilities. The merge must install only the three trained site-specific pallor exports and leave untrained tests unavailable.
5. **The feature UI has a separate camera lifecycle defect.** Earlier UI code rendered one shared ref on multiple video elements, so React retained the final node. The current feature code renders only the active video slot, but its auto-start effect does not depend on `testIndex` or `currentTestResult`; after moving to another camera test, the stream may not reopen. Stream attachment logic is also duplicated. The port should use a single callback ref/effect and make test transitions explicit dependencies.
6. **Required runtime packages were omitted from main.** Pillow is required to decode uploads and `python-multipart` is required for the three-image form. Without the merged dependency lock, image inference or multipart parsing fails before useful output.

### Executable diagnosis results

- All six TorchScript files loaded on CPU, accepted `[1,3,224,224]`, returned finite `[1,2]` logits, and had label counts matching metadata. Artifact corruption is not the cause.
- The corrected current pallor preprocessor produced a finite `[1,3,224,224]` tensor from a 640x480 JPEG.
- On five normal and five risk images per site from the available labeled data, the deployed models classified 29/30 as labeled. This is a smoke check, not an independent clinical evaluation.
- A real multipart request through the FastAPI route returned HTTP 200. The all-normal example produced `normal` with combined risk `0.060400`; the all-risk example produced `risk` with combined risk `0.905903`.
- Both feature inference contract tests passed. They cover single-image routing and three-part multipart aggregation.

### Resolution selected for the merge

- Port the final standards-based Lab conversion and corrected shorter-side resize/center crop, with regression tests for non-square images and train/serve transform equivalence.
- Port only canonical trained `pallor_eye`, `pallor_nail`, and `pallor_palm` runtime pairs. Do not copy generic/placeholder models, backup generations, checkpoints, logs, or datasets.
- Preserve main's physical-input behavior and API contracts; add the vision route and primary risk score as backward-compatible fields.
- Repair camera lifecycle while reconciling the feature UI instead of copying it verbatim.

The reported validation figures are not proof of clinical generalization. Nail and palm datasets contain multiple similarly named images per subject and the trainer splits by image, so train/validation subject leakage is possible. Eye validation accuracy is only 0.626. These limitations must remain visible in the UI/docs; the output is a screening signal, not a diagnosis or hemoglobin measurement.

## Dependency and configuration resolution

- Added Pillow as a direct dependency for upload decoding and `python-multipart` for FastAPI's three-file form parsing, matching the feature branch.
- Added NumPy as a direct dependency because the serving code imports it for CIE Lab conversion; relying on sentence-transformers to install it transitively would make the inference contract fragile.
- Regenerated `uv.lock` without upgrading unrelated packages. The existing CPU-only PyTorch source remains unchanged.
- Frontend packages, environment variables, Dockerfiles, Compose configuration, and nginx/Vite configuration required no substantive merge. The feature copies differ only by line endings in those files.

## Verification

- `79 passed` for the complete backend pytest suite, including main's questionnaire, audio fallback, WHO growth, health, screening/fusion, preprocessing, multipart, and model-registry tests.
- `scripts/verify_vision_inference.py` initialized the installed registry and submitted three generated 640x480 JPEGs through the real multipart FastAPI route. All three MobileNetV3-Large models loaded on CPU; the response contained finite `normal`/`risk` probabilities summing to one, per-site results, and a combined primary risk score. The smoke input classified `normal` with combined risk `0.137951`.
- The same script accepts `--eye`, `--nail`, and `--palm` paths together for repeatable operator checks with real labeled images.
- `npm run build` completed successfully with Vite: 31 modules transformed and production assets emitted.
- `docker compose config --quiet` passed.
- A pre-existing questionnaire regression was found during the full suite: the short answer `no` tied with the opposite fatigue synonym `no stamina`, leaving the integration flow in an endless repeat. A focused scoring guard and regression test fixed it in its own commit.
- TorchScript emits upstream deprecation warnings under the installed future PyTorch version, but loading and inference are successful. Migration to `torch.export` is a future artifact-format change, not required for this merge.

Browser camera permission and physical capture quality require a real kiosk/browser and were not hardware-tested in this environment. The frontend production build and backend multipart contract are verified; the reusable script verifies the same server inference path without camera hardware.

## Merge summary

Pending implementation and verification.
