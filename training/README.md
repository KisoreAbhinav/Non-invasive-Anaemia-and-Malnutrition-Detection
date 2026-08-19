# Model Factory — Vision Model Training Harness

Train and export edge-optimized vision models for the Nourish screening
pipeline.  Drop in your dataset, pick a backbone, and get a deployable
`model.pt` + `model.json` pair.

## Directory Structure

```
training/
├── train.py          # Fine-tuning script
├── export.py         # Export to TorchScript + model.json contract
├── configs/          # Example configs for each vision test
│   ├── pallor.json
│   ├── edema.json
│   └── hair_skin.json
└── README.md
```

## Dataset Format

Use **ImageFolder** layout (one folder per class):

```
data/pallor/
├── normal/
│   ├── img001.jpg
│   └── ...
└── risk/
    ├── img101.jpg
    └── ...
```

## Quick Start

### 1. Train

```bash
cd training
python train.py --config configs/pallor.json
```

### 2. Export

```bash
python export.py --checkpoint checkpoints/pallor/best.pt --test-id pallor --output ../backend/models/vision/pallor
```

### 3. Restart the backend

The model registry picks up new models automatically on startup.

## Configuration

Configs are JSON files with these fields:

| Field | Type | Description |
|---|---|---|
| `test_id` | string | Vision test ID (`pallor`, `edema`, `hair_skin`) |
| `backbone` | string | `mobilenet_v3_small` or `efficientnet_b0` |
| `data_dir` | string | Path to ImageFolder dataset |
| `num_classes` | int | Number of output classes |
| `class_labels` | list[str] | Human-readable class names |
| `primary_score_key` | string | Which class label is the "risk" score |
| `image_size` | int | Input resolution (default 224) |
| `epochs` | int | Training epochs |
| `batch_size` | int | Batch size |
| `learning_rate` | float | Initial learning rate |
| `weight_decay` | float | L2 regularization |
| `freeze_backbone_epochs` | int | Freeze backbone for N epochs (transfer learning) |
| `val_split` | float | Fraction held out for validation |
| `output_activation` | string | `softmax` or `sigmoid` |
| `preprocessing` | object | Normalization params (mean, std, resize, scale) |

## Backbone Comparison

| Backbone | Params | Size (MB) | CPU latency (Pi 5) | Best for |
|---|---|---|---|---|
| MobileNetV3-Small | 2.5M | ~6 | 10–15 ms | Pallor, Edema |
| EfficientNet-B0 | 5.3M | ~16 | 15–22 ms | Hair/Skin |

## Transfer Learning Tips

1. **Freeze backbone first**: Set `freeze_backbone_epochs: 5` to train only
   the classifier head on small datasets (<1000 images).
2. **Unfreeze and fine-tune**: After initial convergence, unfreeze the full
   backbone with a low learning rate (1e-5).
3. **Data augmentation**: The script applies random horizontal flip, rotation
   (±15°), color jitter, and random crop by default.
4. **Class imbalance**: Use `--class-weights balanced` to upweight minority
   classes.

## Export Contract

The export script produces two files:

- `model.pt` — CPU-compatible TorchScript module (`torch.jit.save`)
- `model.json` — Backend drop-in contract (see `backend/models/vision/README.md`)

The contract ensures the model is discovered and executed by the vision
inference pipeline without any code changes.
