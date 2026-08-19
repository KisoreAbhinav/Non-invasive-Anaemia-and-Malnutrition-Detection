# Training 3 Pallor Vision Models — Design Spec

Date: 2026-08-20
Status: Approved
Owner: opencode (via brainstorming skill)

## Context

The Nourish screening pipeline needs 3 trained vision models for pallor detection — one per body site. Each model is a binary classifier (normal vs. risk) deployed to `backend/models/vision/<test_id>/` as a TorchScript `.pt` + `.json` pair.

The training harness (`training/train.py`) and export script (`training/export.py`) already exist. No infrastructure changes are needed — only config files and execution.

## Data

| test_id  | data_dir (training/data/) | normal | risk  | total |
|----------|---------------------------|--------|-------|-------|
| pallor_palm | pallor_palm/           | 1,698  | 2,562 | 4,260 |
| pallor_eye  | pallor_eye/             | 286    | 424   | 710   |
| pallor_nail | pallor_fingernail/ *   | 1,695  | 2,565 | 4,260 |

\* The training data directory is named `pallor_fingernail` but the backend model directory is `pallor_nail`. The export step uses `--test-id pallor_nail` so the model.json gets the correct test_id.

## Architecture

**Backbone:** MobileNetV3-Small for all 3 models (edge-optimized, ~6 MB, matches existing backend contract).

**Training config:**
- Image size: 224×224
- Batch size: 16
- Epochs: 25 (5 frozen backbone + 20 unfrozen)
- Learning rate: 3e-3 initial, 3e-4 after unfreezing
- Weight decay: 1e-4
- Class weights: balanced (normal:risk ≈ 1:1.5)
- Val split: 15% (20% for pallor_eye due to small dataset)
- Augmentation: random horizontal flip, ±15° rotation, color jitter, random crop

## Execution Plan

1. **Create 3 config files** in `training/configs/`:
   - `pallor_palm.json` — data_dir: `data/pallor_palm`, test_id: `pallor_palm`
   - `pallor_eye.json` — data_dir: `data/pallor_eye`, test_id: `pallor_eye`, 20 epochs, 20% val split, lr 1e-3
   - `pallor_nail.json` — data_dir: `data/pallor_fingernail`, test_id: `pallor_nail`

2. **Train each model:**
   ```
   uv run --python backend/.venv python training/train.py --config training/configs/pallor_palm.json
   uv run --python backend/.venv python training/train.py --config training/configs/pallor_eye.json
   uv run --python backend/.venv python training/train.py --config training/configs/pallor_nail.json
   ```
   Best checkpoints saved to `training/checkpoints/<test_id>/best.pt`

3. **Export each model:**
   ```
   uv run --python backend/.venv python training/export.py \
     --checkpoint training/checkpoints/<test_id>/best.pt \
     --test-id <test_id> --backbone mobilenet_v3_small \
     --output backend/models/vision/<test_id>
   ```
   Overwrites placeholder `.pt` and `.json` in backend.

4. **Backup checkpoints:** Best `.pt` files remain in `training/checkpoints/` as the training-side copy.

## Error Handling

- Data dir mismatch already identified and corrected in configs.
- pallor_eye small dataset: 20% val split, lower LR, fewer epochs to reduce overfitting risk.
- If training produces poor val accuracy (<60%), will investigate data quality and hyperparameter adjustments.

## Testing

- Verify backend model registry loads all 3 models correctly.
- Run `test_vision_inference.py` to confirm inference pipeline works.
- Confirm exported model.pt + model.json match the contract.
