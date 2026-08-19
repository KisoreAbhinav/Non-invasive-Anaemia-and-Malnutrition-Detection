# Vision model drop-in contract

Each non-invasive test has its own folder. To install a model, place exactly
these two files in the matching folder and restart the backend:

```text
models/vision/pallor/
├── model.pt
└── model.json
```

`model.pt` must be a CPU-compatible **TorchScript** module saved with
`torch.jit.save`. PyTorch is already the project's CPU inference runtime; no
additional ONNX or TFLite runtime is required. The registry lazily loads the
module with `torch.jit.load(..., map_location="cpu")` on first use and caches it.

`model.json` must follow this contract:

```json
{
  "contract_version": 1,
  "test_id": "pallor",
  "input_shape": [1, 3, 224, 224],
  "preprocessing": {
    "color_space": "RGB",
    "resize": [224, 224],
    "scale": [0.0, 1.0],
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225]
  },
  "output_class_labels": ["normal", "risk"]
}
```

Required keys are `contract_version` (currently `1`), `input_shape` (positive
integer dimensions), `preprocessing` (an object whose details are consumed by
the future test executor), and `output_class_labels` (a non-empty string list).
If present, `test_id` must match the folder name. A missing, partial, or invalid
pair is reported as unavailable and safely skipped; it never prevents startup.

The current model-backed folders are `pallor`, `edema`, and `hair_skin`.
MUAC, child WHO growth indicators, adult/pre-pregnancy BMI, gestational weight,
and fundal height are physical measurements and intentionally do not have model
folders.
