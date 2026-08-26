# Non-invasive Anaemia and Malnutrition Detection

Offline, edge-first anaemia and malnutrition risk screening for children,
pregnant women, and other adults.

The runnable application is in [`project/`](project/). The separate
`Simulation Demo/` is retained as an unrelated edema simulation.

The three runtime vision weights are tracked with Git LFS. Install Git LFS
before cloning, or run `git lfs install && git lfs pull` in an existing clone,
so the application receives the model files rather than pointer files.

```bash
cd project
docker compose up --build
```

Open `http://localhost:8080`. See [`project/README.md`](project/README.md) for
local development, model inference, verification, and Raspberry Pi guidance.
The model/source merge investigation and limitations are documented in
[`MERGE_AUDIT.md`](MERGE_AUDIT.md).
