# Train 3 Pallor Vision Models — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train 3 pallor detection models (eye, nail, palm), save best checkpoints in `training/checkpoints/`, and export them to `backend/models/vision/` to replace placeholders.

**Architecture:** Use existing `training/train.py` and `training/export.py` with 3 new config files. Each model uses MobileNetV3-Small, trained on ImageFolder datasets, saved as TorchScript + JSON contract.

**Tech Stack:** Python 3.13, PyTorch 2.13 (CPU), torchvision 0.28, uv package manager (backend/.venv), PowerShell

---

## File Structure

**Config files to create:**
- `training/configs/pallor_palm.json` — data_dir: `data/pallor_palm`, test_id: `pallor_palm`
- `training/configs/pallor_eye.json` — data_dir: `data/pallor_eye`, test_id: `pallor_eye`, 20 epochs, 20% val split, lr 1e-3
- `training/configs/pallor_nail.json` — data_dir: `data/pallor_fingernail`, test_id: `pallor_nail`

**Checkpoints to create (by train.py):**
- `training/checkpoints/pallor_palm/best.pt`
- `training/checkpoints/pallor_eye/best.pt`
- `training/checkpoints/pallor_nail/best.pt`

**Backend models to overwrite (by export.py):**
- `backend/models/vision/pallor_palm/model.pt` + `model.json`
- `backend/models/vision/pallor_eye/model.pt` + `model.json`
- `backend/models/vision/pallor_nail/model.pt` + `model.json`

## Python Environment

Training uses the backend venv which already has torch + torchvision:
```powershell
uv run --python backend/.venv/Scripts/python.exe
```

---

### Task 1: Create Config Files

**Files:**
- Create: `training/configs/pallor_palm.json`
- Create: `training/configs/pallor_eye.json`
- Create: `training/configs/pallor_nail.json`

- [ ] **Step 1: Create `pallor_palm.json`**

```json
{
  "test_id": "pallor_palm",
  "backbone": "mobilenet_v3_small",
  "data_dir": "data/pallor_palm",
  "num_classes": 2,
  "class_labels": ["normal", "risk"],
  "primary_score_key": "risk",
  "image_size": 224,
  "epochs": 25,
  "batch_size": 16,
  "learning_rate": 3e-3,
  "weight_decay": 1e-4,
  "freeze_backbone_epochs": 5,
  "val_split": 0.15,
  "class_weights": "balanced",
  "output_activation": "softmax",
  "preprocessing": {
    "color_space": "RGB",
    "resize": [224, 224],
    "scale": [0.0, 1.0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

- [ ] **Step 2: Create `pallor_eye.json`** (small dataset → fewer epochs, lower LR, larger val split)

```json
{
  "test_id": "pallor_eye",
  "backbone": "mobilenet_v3_small",
  "data_dir": "data/pallor_eye",
  "num_classes": 2,
  "class_labels": ["normal", "risk"],
  "primary_score_key": "risk",
  "image_size": 224,
  "epochs": 20,
  "batch_size": 16,
  "learning_rate": 1e-3,
  "weight_decay": 1e-4,
  "freeze_backbone_epochs": 5,
  "val_split": 0.20,
  "class_weights": "balanced",
  "output_activation": "softmax",
  "preprocessing": {
    "color_space": "RGB",
    "resize": [224, 224],
    "scale": [0.0, 1.0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

- [ ] **Step 3: Create `pallor_nail.json`**

```json
{
  "test_id": "pallor_nail",
  "backbone": "mobilenet_v3_small",
  "data_dir": "data/pallor_fingernail",
  "num_classes": 2,
  "class_labels": ["normal", "risk"],
  "primary_score_key": "risk",
  "image_size": 224,
  "epochs": 25,
  "batch_size": 16,
  "learning_rate": 3e-3,
  "weight_decay": 1e-4,
  "freeze_backbone_epochs": 5,
  "val_split": 0.15,
  "class_weights": "balanced",
  "output_activation": "softmax",
  "preprocessing": {
    "color_space": "RGB",
    "resize": [224, 224],
    "scale": [0.0, 1.0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  }
}
```

- [ ] **Step 4: Commit config files**

```powershell
git add training/configs/pallor_palm.json training/configs/pallor_eye.json training/configs/pallor_nail.json
git commit -m "train: add configs for pallor_palm, pallor_eye, pallor_nail"
```

---

### Task 2: Train pallor_palm Model

**Files:**
- Input: `training/configs/pallor_palm.json`, `training/data/pallor_palm/`
- Output: `training/checkpoints/pallor_palm/best.pt`

- [ ] **Step 1: Run training**

```powershell
cd training
uv run --python ..\backend\.venv\Scripts\python.exe python train.py --config configs/pallor_palm.json
```
Expected: 25 epochs, logs show `train_acc` and `val_acc` per epoch, best checkpoint saved to `checkpoints/pallor_palm/best.pt`

- [ ] **Step 2: Verify checkpoint exists**

```powershell
Test-Path training/checkpoints/pallor_palm/best.pt
```
Expected: `True`

- [ ] **Step 3: Record best val_acc from training output**

Look for: `Training complete. Best val_acc=0.XXX  Saved to checkpoints/pallor_palm/best.pt`

- [ ] **Step 4: Commit** (no code change — checkpoints dir is .gitignored or kept local only)

---

### Task 3: Train pallor_eye Model

**Files:**
- Input: `training/configs/pallor_eye.json`, `training/data/pallor_eye/`
- Output: `training/checkpoints/pallor_eye/best.pt`

- [ ] **Step 1: Run training**

```powershell
cd training
uv run --python ..\backend\.venv\Scripts\python.exe python train.py --config configs/pallor_eye.json
```
Expected: 20 epochs, logs show training metrics, best checkpoint saved to `checkpoints/pallor_eye/best.pt`

- [ ] **Step 2: Verify checkpoint exists**

```powershell
Test-Path training/checkpoints/pallor_eye/best.pt
```
Expected: `True`

- [ ] **Step 3: Record best val_acc**

---

### Task 4: Train pallor_nail Model

**Files:**
- Input: `training/configs/pallor_nail.json`, `training/data/pallor_fingernail/`
- Output: `training/checkpoints/pallor_nail/best.pt`

- [ ] **Step 1: Run training**

```powershell
cd training
uv run --python ..\backend\.venv\Scripts\python.exe python train.py --config configs/pallor_nail.json
```
Expected: 25 epochs, logs show training metrics, best checkpoint saved to `checkpoints/pallor_nail/best.pt`

- [ ] **Step 2: Verify checkpoint exists**

```powershell
Test-Path training/checkpoints/pallor_nail/best.pt
```
Expected: `True`

- [ ] **Step 3: Record best val_acc**

---

### Task 5: Export pallor_palm Model to Backend

**Files:**
- Input: `training/checkpoints/pallor_palm/best.pt`
- Output: `backend/models/vision/pallor_palm/model.pt` + `model.json`

- [ ] **Step 1: Run export**

```powershell
cd training
uv run --python ..\backend\.venv\Scripts\python.exe python export.py `
  --checkpoint checkpoints/pallor_palm/best.pt `
  --test-id pallor_palm `
  --backbone mobilenet_v3_small `
  --num-classes 2 `
  --class-labels normal risk `
  --primary-score-key risk
```
Expected: `Saved TorchScript module: .../pallor_palm/model.pt (X.X MB)` and `Saved contract: .../pallor_palm/model.json`

- [ ] **Step 2: Verify exported files**

```powershell
Test-Path backend/models/vision/pallor_palm/model.pt
Test-Path backend/models/vision/pallor_palm/model.json
```
Expected: both `True`

- [ ] **Step 3: Verify model.json contract**

```powershell
Get-Content backend/models/vision/pallor_palm/model.json
```
Expected: `"test_id": "pallor_palm"`, `"output_activation": "softmax"`, `"primary_score_key": "risk"`

---

### Task 6: Export pallor_eye Model to Backend

**Files:**
- Input: `training/checkpoints/pallor_eye/best.pt`
- Output: `backend/models/vision/pallor_eye/model.pt` + `model.json`

- [ ] **Step 1: Run export**

```powershell
cd training
uv run --python ..\backend\.venv\Scripts\python.exe python export.py `
  --checkpoint checkpoints/pallor_eye/best.pt `
  --test-id pallor_eye `
  --backbone mobilenet_v3_small `
  --num-classes 2 `
  --class-labels normal risk `
  --primary-score-key risk
```

- [ ] **Step 2: Verify exported files** (same as Task 5 Step 2/3, with `pallor_eye`)

---

### Task 7: Export pallor_nail Model to Backend

**Files:**
- Input: `training/checkpoints/pallor_nail/best.pt`
- Output: `backend/models/vision/pallor_nail/model.pt` + `model.json`

- [ ] **Step 1: Run export**

```powershell
cd training
uv run --python ..\backend\.venv\Scripts\python.exe python export.py `
  --checkpoint checkpoints/pallor_nail/best.pt `
  --test-id pallor_nail `
  --backbone mobilenet_v3_small `
  --num-classes 2 `
  --class-labels normal risk `
  --primary-score-key risk
```

- [ ] **Step 2: Verify exported files** (same as Task 5 Step 2/3, with `pallor_nail`)

---

### Task 8: Verify Backend Inference

**Files:**
- Test: `backend/tests/test_vision_inference.py`

- [ ] **Step 1: Run tests**

```powershell
cd backend
uv run --python .venv/Scripts/python.exe python -m pytest tests/test_vision_inference.py -v
```
Expected: Both tests pass (`test_camera_image_runs_registered_model_and_returns_named_scores` and `test_multipart_sub_captures_runs_all_models_and_aggregates`)

- [ ] **Step 2: Manual smoke test** (optional, if tests fail or for extra confidence)

Start the backend and send an image:
```powershell
cd backend
uv run --python .venv/Scripts/python.exe python -m uvicorn app.main:app --reload
```
Then POST a real palm image to `/api/flows/screening/vision/pallor?population=child_under5` with `pallor_eye`, `pallor_nail`, `pallor_palm` file fields.

Expected: `200 OK`, JSON response with `test_id: "pallor"`, `per_image` array of 3, and `scores.risk` > `scores.normal` for anemic imagery.
