# HANDOFF REPORT — Camera Capture for Pallor Tests

## What This Project Is

**Nourish** — an edge-first (Raspberry Pi 5) non-invasive anemia and malnutrition screening system. Voice-led kiosk that runs offline. 4 stages: Questionnaire → Visual Cues (camera + physical measurements) → Clinical Results → Final Verdict.

**Stack**: FastAPI backend, React (Vite) frontend, PyTorch TorchScript models, 800×480px fixed viewport (kiosk display).

## What Needs To Be Fixed (THE CAMERA ISSUE)

### The Goal
The pallor test requires **3 separate camera captures** — one for the eye (conjunctiva), one for the nail bed, and one for the palm. The UI should show 3 black boxes side by side. The camera feed appears in box 1, user hits Capture, the frame freezes on box 1, the feed moves to box 2, and so on. After all 3 are captured, user hits "Run inference" to send all 3 to the backend.

### What's Broken
1. **Camera doesn't open reliably** — the `getUserMedia` call works sometimes but the video element often doesn't display the stream
2. **Feed appears on the wrong box** — when all 3 slots are empty, every slot renders a `<video ref={videoRef}>` and React assigns the ref to the LAST video element (slot 2), so the stream always attaches to the last box
3. **Layout shifts on capture** — when a captured image replaces the video in a slot, the box expands/contracts because the image or video has intrinsic dimensions that aren't constrained
4. **The execution card header was broken** — I added camera/upload icon buttons to the header which broke the original card layout. This was partially reverted but may still have issues

### Root Cause Analysis

The fundamental problem is how React refs work with dynamic lists. When I render:
```jsx
{subCaps.map((cap, idx) => (
  <div key={cap.id}>
    {capturedImages[idx] ? (
      <img ... />
    ) : (
      <video ref={videoRef} ... />  // <-- BUG: ALL empty slots get a video
    )}
  </div>
))}
```

React sets `videoRef.current` to the **last** rendered `<video>` element. With 3 empty slots, that's slot 2, not slot 0.

### The Fix That's Needed

**Option A (recommended)**: Only render the `<video>` element in the **active** slot. Other empty slots show a dark placeholder. This is what the current code attempts but something is still broken — likely a timing issue with the auto-open camera effect and the stream attachment useEffect.

**Option B**: Use `react-webcam` package which handles all of this internally. It provides `<Webcam>` component with `getScreenshot()` method. This would be much simpler.

**Option C**: Keep the `<video>` element OUTSIDE the slot grid entirely (in a fixed/hidden position), and use canvas `drawImage` to freeze frames into each slot. The slots would only ever contain `<img>` or empty placeholders, never `<video>`.

### My Recommendation
Use **react-webcam** (`npm install react-webcam`). It handles stream lifecycle, error states, and screenshots cleanly. Replace the video/canvas/ref mess with:
```jsx
import Webcam from "react-webcam";
<Webcam ref={webcamRef} screenshotFormat="image/jpeg" />
// Capture: webcamRef.current.getScreenshot() returns base64
```

## Current File State

### Frontend

**`frontend/src/App.jsx`** — The main (and only) React component. ~1400 lines.

Key camera-related state:
```js
const [capturedImages, setCapturedImages] = useState([]);      // Array of Blobs
const [capturedPreviewUrls, setCapturedPreviewUrls] = useState([]); // Array of object URLs
const [activeSlot, setActiveSlot] = useState(0);                // Which slot gets the camera
const cameraStreamRef = useRef(null);                           // The MediaStream
const videoRef = useRef(null);                                  // Video element ref
const canvasRef = useRef(null);                                 // Hidden canvas for capture
const [capturedImage, setCapturedImage] = useState(null);      // Single image mode (non-pallor)
const [capturedImageUrl, setCapturedImageUrl] = useState(null);
```

Key camera functions:
- `closeCamera()` — stops stream tracks, clears ref
- `startCamera()` — calls `getUserMedia`, stores stream in ref (effect attaches to video)
- `captureToSlot(slotIndex, closeAfter)` — draws video frame to canvas (center-cropped to square), gets JPEG blob, stores in `capturedImages[slotIndex]`, advances `activeSlot`
- `capturePhoto()` — single-image capture wrapper
- `handleFileUpload()` / `handleFileUploadToSlot()` — file input handlers
- `clearSlot(idx)` — removes a captured image from a slot

Key effects:
- Auto-open camera effect (line ~833): when entering a camera test, calls `startCamera()` after 200ms delay
- Stream attachment effect (line ~536): runs on every render, attaches `cameraStreamRef.current` to `videoRef.current` if they differ

The `renderTest()` function (line ~1188) renders:
- For tests with `sub_captures` (pallor): 3-slot canvas + single capture button
- For other camera tests: single video + capture/cancel buttons
- For physical tests: measurement input grid

**`frontend/src/index.css`** — All styles. Key classes:
- `.sub-capture-canvas` — grid of 3 slots, `height: 200px`
- `.sub-slot` — flex column, `overflow: hidden`, `height: 100%`
- `.sub-slot-camera` / `.sub-slot-video` — video fills slot absolutely
- `.sub-slot-image` / `.sub-slot-image img` — captured image fills slot absolutely

### Backend

**`backend/app/flows/screening.py`** — API endpoints:
- `POST /api/flows/screening/vision/{test_id}?population=X` — accepts single image (Content-Type: image/*) or multipart form-data for tests with `sub_captures`
- The multipart handler reads files by `sub_capture.id` keys (e.g., `pallor_eye`, `pallor_nail`, `pallor_palm`)

**`backend/app/flows/vision_inference.py`** — Inference:
- `predict_image(test_id, image_bytes)` — single image inference
- `predict_batch(sub_captures, image_bytes_list)` — multi-image: runs each through its model, averages risk scores, returns combined result with `per_image` breakdown

**`backend/config/visual-cues.default.json`** — Test definitions:
```json
{
  "id": "pallor",
  "name": "Conjunctival, nail & palm pallor",
  "sub_captures": [
    { "id": "pallor_eye", "label": "EYE", "model_dir": "pallor_eye" },
    { "id": "pallor_nail", "label": "NAILBED", "model_dir": "pallor_nail" },
    { "id": "pallor_palm", "label": "PALM", "model_dir": "pallor_palm" }
  ]
}
```

**`backend/models/vision/`** — Model directories:
- `pallor_eye/model.pt` + `model.json` (MobileNetV3-Small, 6.2MB)
- `pallor_nail/model.pt` + `model.json` (MobileNetV3-Small, 6.2MB)
- `pallor_palm/model.pt` + `model.json` (MobileNetV3-Small, 6.2MB)
- `edema/model.pt` + `model.json` (MobileNetV3-Small, 6.2MB)
- `hair_skin/model.pt` + `model.json` (EfficientNet-B0, 16MB)

**⚠️ IMPORTANT**: These models are ImageNet pre-trained weights with swapped classifier heads. They have NOT been trained on medical data. They will give ~50/50 random predictions. The user has datasets and needs to train them using the `training/` harness.

**`backend/scripts/setup_vision_models.py`** — Builds the placeholder models from torchvision pretrained weights.

### Training Harness

**`training/`** — Model factory:
- `train.py` — Fine-tuning script (ImageFolder format, MobileNetV3-Small or EfficientNet-B0)
- `export.py` — Exports trained checkpoint to TorchScript + model.json contract
- `configs/pallor.json`, `edema.json`, `hair_skin.json` — Example configs
- `data/pallor_eye/{normal,risk}/`, `data/pallor_nail/{normal,risk}/`, `data/pallor_palm/{normal,risk}/` — Empty directories ready for data

## How The Flow Works

1. User starts screening → questionnaire → completes or skips
2. Plan is built → pallor test shows with `sub_captures: [pallor_eye, pallor_nail, pallor_palm]`
3. User enters the test → camera auto-opens → 3 black boxes appear
4. Camera feed in box 1 → user captures → frame freezes on box 1 → feed moves to box 2 → etc.
5. All 3 captured → user hits "Run inference" (Enter key)
6. Frontend sends multipart POST with 3 images to `/api/flows/screening/vision/pallor?population=X`
7. Backend runs each image through its respective model, averages scores
8. Results shown: per-image breakdown + combined classification
9. Continues to next test (edema, muac, etc.)

## Known Issues Beyond Camera

1. The `visual_results` sent to the final `/result` endpoint need to include the `score` field for sensor fusion to work (this was added to `VisualResult` model)
2. The `advanceTest` function stores visual results before the vision test actually runs — the `recorded` object is created with `value: null` for camera tests, then `submitVisionTest` overwrites it. This double-write is fragile.
3. The `activeSlot` state can get out of sync if the user navigates away and back
4. The auto-open camera effect has a 200ms delay which may not be enough on slow devices

## Git State

Branch: `feat/model-inference`
Modified files (unstaged):
- `backend/app/flows/model_registry.py`
- `backend/app/flows/screening.py`
- `backend/pyproject.toml`
- `backend/tests/test_screening_flow.py`
- `backend/uv.lock`

Untracked files:
- `backend/app/flows/vision_inference.py`
- `backend/tests/test_vision_inference.py`
- `training/` (entire directory)

## How To Run

```bash
# Backend
cd backend
.venv/Scripts/python.exe -m uvicorn app.main:app --reload

# Frontend
cd frontend
npm run dev

# Build models (already done)
cd backend
.venv/Scripts/python.exe scripts/setup_vision_models.py

# Tests
cd backend
.venv/Scripts/python.exe -m pytest tests/ -v
```

## What The Next Agent Should Do

1. **Fix the camera** — either use react-webcam or fix the native API approach. The core issue is ref assignment in dynamic lists and stream attachment timing.
2. **Fix layout shifts** — ensure the 3 slots stay at fixed height when content changes from video to image
3. **Test the full flow** end-to-end: camera opens → 3 captures → inference → results
4. **Verify the backend multipart handler** works with the frontend FormData submission
5. Eventually the user will need to train the models — the `training/` harness is ready for that
