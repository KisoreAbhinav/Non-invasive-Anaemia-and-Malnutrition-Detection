# Implementation status

## Completed

- [x] Created an independent `Simulation Demo` FastAPI/browser-canvas project without changing the reference device project.
- [x] Implemented a shared parameterized normal/edema press-and-recovery physics model with residual edema dent and configurable deterministic detection.
- [x] Built the interactive fabric mesh, click-to-position press point, sliders, live curve, metrics, and deterministic verdict.
- [x] Added synthetic labelled patient generation, illustrative noisy lab panels, JSONL persistence, in-app record table, and export.
- [x] Added a lightweight synthetic camera-frame feature classifier and a train/compare UI.
- [x] Added safety disclaimers, reference notes, project README, ignore rules, and focused automated tests.
- [x] Verified the API health, normal/edema deterministic outcomes, dataset generation/export, classifier training, Python compilation, browser JavaScript syntax, and automated test suite (3 passed).
- [x] Fixed editable-install package discovery by explicitly packaging only the Python `app` module; verified a clean virtual-environment install and test run (3 passed).
- [x] Consolidated the UI into one page: selected tissue mode now generates one synthetic patient and immediately simulates that patient’s generated force, duration, and tissue parameters.
- [x] Removed the tabbed dataset/classifier workflow and prominent disclaimer copy; vision-model training remains as one compact action on the same page.
- [x] Re-themed the page with the exact core colour tokens used by the reference frontend (`#efede3`, `#e2dfd5`, `#f7f4e9`, charcoal, olive, and ochre).
- [x] Re-verified the generated-patient → parameterized simulation path, JavaScript syntax, and automated tests (3 passed).
- [x] Added user-controlled force and press-duration sliders; their selected values are persisted in the generated patient record and drive that exact simulation.
- [x] Added full generated-patient Albumin, Hemoglobin, and blood-glucose profiles with causal-link, mechanism, and edema-expected-value details; edema-mode Albumin is constrained below 3.0 g/dL.
- [x] Added a light bordered `VISION ROI` square at the selected impact point during the simulation.
- [x] Re-verified selected force/duration, edema-mode Albumin, parameterized simulation, JavaScript syntax, and automated tests (4 passed).
- [x] Renamed the demo, simplified the compact patient card to Hb, Albumin, and blood glucose, and removed the vision-model training controls.
- [x] Replaced the simulation duration metric and vision verdict with delayed visual pit detection: the status scans first, then draws a large ROI only while an indentation remains; normal tissue returns to scanning after rebound.
- [x] Made the detected ROI contract with the simulated indentation during recovery, and clear the current detection when the press point is moved.
- [x] Replaced the depth-driven ROI with baseline frame differencing and connected-region detection on the rendered synthetic camera frame; the detected region now determines the ROI location and size.

## Remaining

- [x] No implementation tasks remain.
