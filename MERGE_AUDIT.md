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

Pending executable reproduction and targeted verification in Phase 2.

## Merge summary

Pending implementation and verification.
