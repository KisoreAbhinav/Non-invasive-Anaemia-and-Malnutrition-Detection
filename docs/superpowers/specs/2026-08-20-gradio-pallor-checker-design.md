# Gradio Pallor Checker — Design

## Goal

A single-page Gradio interface that runs in the `training/` environment and lets a user upload images for all three pallor checks (Palm, Eye, Nail) and see each result — class badge + confidence — on one page. It uses the already-deployed models (`backend/models/vision/pallor_palm`, `pallor_eye`, `pallor_nail`) so after retraining, the dashboard picks up the same artifacts the real backend serves.

## Scope

In scope:
- `training/gradio_app.py` — the only new file.
- Installing `gradio` into `training/.venv` (existing env, pip-based; no uv migration).

Out of scope:
- Changes to backend code, model contracts, or training pipeline.
- Combined/averaged verdict across sites (user chose separate cards).
- Model hot-reload (models load once at startup; restart to pick up retrained models).

## Architecture

Single standalone script. No imports from the `backend/app` package. All preprocessing is contract-driven: read the `preprocessing` block from each `model.json` and apply it exactly as the backend does, so there is no train/serve drift even if a model is re-exported with different settings.

Each model directory is loaded once at startup into an in-memory struct:
- `model.pt` → `torch.jit.load(..., map_location="cpu")`
- `model.json` → parsed contract (input shape, preprocessing, labels, primary score key)

If a directory is missing either file, that card is marked "model not installed" with a warning logged at startup; the other cards keep working.

## Preprocessing (matches backend / train_all_v2 exactly)

For each uploaded image, on the raw RGB image:

1. `PIL.Image.open(bytes).convert("RGB")` (reject unreadable bytes with a per-card friendly error).
2. Resize to `preprocessing["resize"]` (256×256) preserving aspect ratio (shorter side scaled, bilinear) — same as the backend's `center_crop` path.
3. Center-crop to `preprocessing["center_crop"]` (224×224), same crop math as torchvision `CenterCrop`.
4. Convert to CIE Lab via the exact numpy sRGB→linear→XYZ(D65)→Lab math used in `train_all_v2.py:LabTensor` (PIL's `Image.convert("LAB")` is deliberately NOT used):
   - L → `/ 100`
   - a,b → `clip((val + 128) / 255, 0, 1)`
   - Result float tensor in `[0, 1]`, shape `(1, 3, 224, 224)`.
5. Normalize with the contract's `mean`/`std` (`[0.5,0.5,0.5]` / `[0.5,0.25,0.25]`), read from `model.json`, not hardcoded.

The Lab conversion helpers are copied into `gradio_app.py` (same constants and functions as the backend) and are identical in behavior — verified numerically to float rounding.

## UI

One page, three cards laid out in a row (wrapping on narrow screens):

- **Palm** — upload → result
- **Eye** — upload → result
- **Nail** — upload → result

Each card contains:
- A title (e.g. "Palm palmar pallor").
- A Gradio image upload component (`gr.Image(type="filepath")`).
- A result badge rendered as colored Markdown/HTML: red for **Risk**, green for **Normal**, plus the confidence percentage of the primary score key (e.g. **Risk 88%**). Confidence is the softmax probability of the `primary_score_key` ("risk").
- A small muted line showing the active model file: `model: backend\models\vision\pallor_palm\model.pt` (absolute path).

## Error Handling

- Unreadable/invalid upload → red message in that card; other cards unaffected.
- Missing model files at startup → warning log + "model not installed" in that card's result area.
- Framework errors per prediction are caught per card so one failing card never takes down the page.

## Startup & Running

```
& .venv/Scripts/python.exe gradio_app.py          # training/ dir, default port 7860
& .venv/Scripts/python.exe gradio_app.py --port 7861
```

`gr.Blocks` (not `gr.Interface`) since three independent input/output pairs are needed. Launch with `block.launch()`; no admin/server extras.

## Testing

- Install gradio, then run `gradio_app.py --smoke` which:
  - loads all three model directories (asserts each card has a usable model or logs the failure);
  - runs one prediction each on Palm, Eye, Nail sample images from `training/data/**` and prints the badge + confidence.
- A quick numeric check comparing the app's Lab conversion against `train_all_v2.py`'s `LabTensor` on the same image (max abs diff ≈ 1e-7).
- Manual check: open the page, upload one image per site, confirm badges/confidences appear per card.

## Success Criteria

- Page starts from the training venv with no backend process running.
- All three cards classify an uploaded image and show badge + confidence + model path.
- A retrained model (re-run of `train_all_v2.py`) is reflected in the page after restarting the app.