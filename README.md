# Non-invasive Anaemia and Malnutrition Detection

Offline, edge-first anaemia and malnutrition risk screening for children,
pregnant women, and other adults.

The runnable application is in [`project/`](project/). The separate
`Simulation Demo/` is retained as an unrelated edema simulation.

```bash
cd project
docker compose up --build
```

Open `http://localhost:8080`. See [`project/README.md`](project/README.md) for
local development, model inference, verification, and Raspberry Pi guidance.
The model/source merge investigation and limitations are documented in
[`MERGE_AUDIT.md`](MERGE_AUDIT.md).
