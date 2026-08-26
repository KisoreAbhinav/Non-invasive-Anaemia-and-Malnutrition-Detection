const $ = (selector) => document.querySelector(selector);
const fabric = $("#fabric"), fabricCtx = fabric.getContext("2d"), graph = $("#graph"), graphCtx = graph.getContext("2d");

let pressPoint = { x: .5, y: .52 };
let latest = null, patient = null, playbackStart = 0, animationId = 0, simulationRunId = 0;
let visionBaseline = null, visionDetection = null, visionDetected = false, lastVisionScan = 0;

const VISION_SCAN_DELAY = .8;
const VISION_SCAN_INTERVAL = 110;

function resizeCanvas(canvas) {
  const ratio = devicePixelRatio || 1, box = canvas.getBoundingClientRect();
  const pixelWidth = Math.floor(box.width * ratio), pixelHeight = Math.floor(box.height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) { canvas.width = pixelWidth; canvas.height = pixelHeight; }
  return { w: pixelWidth / ratio, h: pixelHeight / ratio, ratio };
}

function depthAt(series, time) { return series.reduce((best, point) => Math.abs(point.t - time) < Math.abs(best.t - time) ? point : best).depth; }
function setModeBadge() { const edema = $("#mode").value === "edema"; $("#modeBadge").textContent = edema ? "Edema tissue" : "Normal tissue"; $("#modeBadge").classList.toggle("edema", edema); }
function updatePressControls() { $("#forceValue").textContent = (+$("#force").value).toFixed(1); $("#durationValue").textContent = `${(+$("#duration").value).toFixed(1)} s`; }

function setVisionStatus(detected, detail) {
  visionDetected = detected;
  $("#visionStatus").textContent = detected ? "Pit detected" : "Looking for pit";
  $("#visionStatusReason").textContent = detail;
}

function resetVisionDetector(detail = "Awaiting simulation") {
  visionBaseline = null;
  visionDetection = null;
  lastVisionScan = 0;
  setVisionStatus(false, detail);
}

function drawDentShading(depth, w, h, px, py) {
  if (!latest || depth <= 0) return;
  const strength = Math.max(0, Math.min(depth / latest.simulation.peak_depth, 1));
  if (strength < .015) return;

  // The rendered concavity gets shallower and narrower as it rebounds. The
  // vision path below measures only these camera-frame pixels.
  const radius = Math.min(w, h) * (.055 + .16 * Math.sqrt(strength));
  const shadow = fabricCtx.createRadialGradient(px, py, radius * .05, px, py, radius);
  shadow.addColorStop(0, `rgba(8,8,7,${(.56 * strength).toFixed(3)})`);
  shadow.addColorStop(.52, `rgba(18,18,16,${(.34 * strength).toFixed(3)})`);
  shadow.addColorStop(1, "rgba(39,39,36,0)");
  fabricCtx.fillStyle = shadow;
  fabricCtx.beginPath(); fabricCtx.arc(px, py, radius, 0, Math.PI * 2); fabricCtx.fill();

  const highlight = fabricCtx.createRadialGradient(px - radius * .34, py - radius * .34, 0, px - radius * .34, py - radius * .34, radius * .55);
  highlight.addColorStop(0, `rgba(247,244,233,${(.16 * strength).toFixed(3)})`);
  highlight.addColorStop(1, "rgba(247,244,233,0)");
  fabricCtx.fillStyle = highlight;
  fabricCtx.beginPath(); fabricCtx.arc(px - radius * .34, py - radius * .34, radius * .55, 0, Math.PI * 2); fabricCtx.fill();
}

function drawDetectionBox() {
  if (!visionDetection) return;
  const { x, y, size } = visionDetection;
  fabricCtx.fillStyle = "rgba(247,244,233,.08)";
  fabricCtx.fillRect(x - size / 2, y - size / 2, size, size);
  fabricCtx.strokeStyle = "rgba(247,244,233,.96)";
  fabricCtx.lineWidth = 3;
  fabricCtx.strokeRect(x - size / 2, y - size / 2, size, size);
  fabricCtx.fillStyle = "#f7f4e9";
  fabricCtx.font = "700 10px system-ui";
  fabricCtx.fillText("VISION: PIT DETECTED", x - size / 2, y - size / 2 - 8);
}

function drawFabric(depth = 0, includeDetection = true) {
  const { w, h, ratio } = resizeCanvas(fabric);
  fabricCtx.setTransform(ratio, 0, 0, ratio, 0, 0);
  const bg = fabricCtx.createLinearGradient(0, 0, w, h);
  bg.addColorStop(0, "#686762"); bg.addColorStop(1, "#272724");
  fabricCtx.fillStyle = bg; fabricCtx.fillRect(0, 0, w, h);

  const px = pressPoint.x * w, py = pressPoint.y * h, cols = 25, rows = 17;
  drawDentShading(depth, w, h, px, py);
  const project = (x, y) => {
    const dx = (x - px) / (w * .27), dy = (y - py) / (h * .29);
    const dent = Math.exp(-(dx * dx + dy * dy) * 3) * depth;
    return { x, y: y + dent * h * .42 };
  };

  fabricCtx.lineWidth = 1; fabricCtx.strokeStyle = "rgba(247,244,233,.64)";
  for (let row = 0; row <= rows; row++) {
    fabricCtx.beginPath();
    for (let col = 0; col <= cols; col++) { const p = project(col * w / cols, row * h / rows); col ? fabricCtx.lineTo(p.x, p.y) : fabricCtx.moveTo(p.x, p.y); }
    fabricCtx.stroke();
  }
  for (let col = 0; col <= cols; col++) {
    fabricCtx.beginPath();
    for (let row = 0; row <= rows; row++) { const p = project(col * w / cols, row * h / rows); row ? fabricCtx.lineTo(p.x, p.y) : fabricCtx.moveTo(p.x, p.y); }
    fabricCtx.stroke();
  }

  const glowRadius = 10 + depth * 17;
  const glow = fabricCtx.createRadialGradient(px - glowRadius * .3, py - glowRadius * .3, 1, px, py, glowRadius * 1.8);
  glow.addColorStop(0, "rgba(247,244,233,.35)"); glow.addColorStop(1, "rgba(39,39,36,0)");
  fabricCtx.fillStyle = glow; fabricCtx.beginPath(); fabricCtx.arc(px, py, glowRadius * 1.8, 0, Math.PI * 2); fabricCtx.fill();
  if (includeDetection && visionDetected) drawDetectionBox();

  $("#meshHint").style.display = latest ? "block" : "none";
  $("#meshHint").style.left = `${pressPoint.x * 100}%`;
  $("#meshHint").style.top = `${pressPoint.y * 100}%`;
}

function luminance(data, width, height, x, y, radius) {
  let total = 0, samples = 0;
  for (const offsetY of [-radius, 0, radius]) for (const offsetX of [-radius, 0, radius]) {
    const px = Math.max(0, Math.min(width - 1, Math.round(x + offsetX))), py = Math.max(0, Math.min(height - 1, Math.round(y + offsetY)));
    const index = (py * width + px) * 4;
    total += .2126 * data[index] + .7152 * data[index + 1] + .0722 * data[index + 2];
    samples += 1;
  }
  return total / samples;
}

function captureVisionBaseline() {
  visionBaseline = {
    width: fabric.width,
    height: fabric.height,
    data: fabricCtx.getImageData(0, 0, fabric.width, fabric.height).data,
  };
}

function detectPitFromFrame() {
  if (!visionBaseline || visionBaseline.width !== fabric.width || visionBaseline.height !== fabric.height) return null;
  const frame = fabricCtx.getImageData(0, 0, fabric.width, fabric.height);
  const { data, width, height } = frame, ratio = fabric.width / fabric.getBoundingClientRect().width;
  const step = Math.max(6, Math.round(6 * ratio)), blurRadius = Math.max(1, Math.round(step / 4));
  const cols = Math.floor((width - step) / step), rows = Math.floor((height - step) / step);
  const values = new Float32Array(cols * rows), threshold = 4.5;

  for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) {
    const x = step + col * step, y = step + row * step, index = row * cols + col;
    values[index] = luminance(visionBaseline.data, width, height, x, y, blurRadius) - luminance(data, width, height, x, y, blurRadius);
  }

  const visited = new Uint8Array(values.length);
  let best = null;
  for (let start = 0; start < values.length; start++) {
    if (visited[start] || values[start] < threshold) continue;
    const queue = [start]; visited[start] = 1;
    let count = 0, weight = 0, weightedX = 0, weightedY = 0, minCol = cols, maxCol = 0, minRow = rows, maxRow = 0;
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const index = queue[cursor], row = Math.floor(index / cols), col = index % cols, value = values[index];
      count += 1; weight += value; weightedX += (step + col * step) * value; weightedY += (step + row * step) * value;
      minCol = Math.min(minCol, col); maxCol = Math.max(maxCol, col); minRow = Math.min(minRow, row); maxRow = Math.max(maxRow, row);
      for (const [rowOffset, colOffset] of [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 1], [1, -1], [1, 0], [1, 1]]) {
        const nextRow = row + rowOffset, nextCol = col + colOffset, next = nextRow * cols + nextCol;
        if (nextRow >= 0 && nextRow < rows && nextCol >= 0 && nextCol < cols && !visited[next] && values[next] >= threshold) { visited[next] = 1; queue.push(next); }
      }
    }
    const boxCols = maxCol - minCol + 1, boxRows = maxRow - minRow + 1, density = count / (boxCols * boxRows);
    if (count < 12 || Math.min(boxCols, boxRows) < 3 || density < .24) continue;
    const score = weight * Math.sqrt(count) * density;
    if (!best || score > best.score) best = { score, weight, weightedX, weightedY, boxCols, boxRows };
  }

  if (!best) return null;
  const footprint = Math.max(best.boxCols, best.boxRows) * step / ratio;
  return {
    x: best.weightedX / best.weight / ratio,
    y: best.weightedY / best.weight / ratio,
    size: Math.max(56, footprint * 1.35),
    confidence: Math.min(.99, best.score / 6500),
  };
}

function scanForPit(now, visualTime) {
  const releaseAt = latest.simulation.duration + VISION_SCAN_DELAY;
  if (visualTime < releaseAt) { setVisionStatus(false, "Capturing baseline and scanning"); return false; }
  if (now - lastVisionScan < VISION_SCAN_INTERVAL) return false;
  lastVisionScan = now;
  const detection = detectPitFromFrame();
  if (detection) {
    visionDetection = detection;
    setVisionStatus(true, `Concavity found in camera frame · ${Math.round(detection.confidence * 100)}%`);
  } else {
    visionDetection = null;
    setVisionStatus(false, "No persistent concavity in camera frame");
  }
  return true;
}

function drawGraph(series, duration) {
  const { w, h, ratio } = resizeCanvas(graph); graphCtx.setTransform(ratio, 0, 0, ratio, 0, 0); graphCtx.clearRect(0, 0, w, h);
  const pad = { l: 42, r: 15, t: 15, b: 28 }, iw = w - pad.l - pad.r, ih = h - pad.t - pad.b, maxT = series.at(-1).t, maxD = Math.max(...series.map(p => p.depth)) * 1.15 || 1, x = t => pad.l + t / maxT * iw, y = d => pad.t + ih - d / maxD * ih;
  graphCtx.fillStyle = "#f7f4e9"; graphCtx.fillRect(0, 0, w, h); graphCtx.strokeStyle = "#d4d1c8"; graphCtx.lineWidth = 1;
  for (let i = 0; i < 5; i++) { const lineY = pad.t + ih * i / 4; graphCtx.beginPath(); graphCtx.moveTo(pad.l, lineY); graphCtx.lineTo(w - pad.r, lineY); graphCtx.stroke(); }
  graphCtx.fillStyle = "#5f5e59"; graphCtx.font = "11px system-ui"; [0, Math.round(maxT / 2), Math.round(maxT)].forEach(t => graphCtx.fillText(`${t}s`, x(t) - 7, h - 9));
  graphCtx.fillStyle = "rgba(138,105,46,.14)"; graphCtx.fillRect(x(0), pad.t, x(duration), ih); graphCtx.fillStyle = "rgba(72,107,81,.08)"; graphCtx.fillRect(x(duration), pad.t, w - pad.r - x(duration), ih);
  graphCtx.strokeStyle = "#486b51"; graphCtx.lineWidth = 2.4; graphCtx.beginPath(); series.forEach((point, index) => index ? graphCtx.lineTo(x(point.t), y(point.depth)) : graphCtx.moveTo(x(point.t), y(point.depth))); graphCtx.stroke(); graphCtx.fillStyle = "#8a692e"; graphCtx.fillRect(x(duration) - 1, pad.t, 2, ih);
}

function showPatient(record) {
  const values = record.blood_values, isEdema = record.label.tissue_mode === "edema";
  $("#patientCard").className = `patient-card ${isEdema ? "edema" : ""}`;
  $("#patientCard").innerHTML = `<span class="patient-label">${isEdema ? "Edema-mode" : "Normal-mode"} synthetic case</span><div class="patient-grid"><div><span>Hb</span><strong>${values.hemoglobin_g_dL} g/dL</strong></div><div><span>Albumin</span><strong>${values.albumin_g_dL} g/dL</strong></div><div><span>Blood glucose</span><strong>${values.blood_glucose_mg_dL} mg/dL</strong></div></div>`;
  $("#simulationTitle").textContent = `${isEdema ? "Edema" : "Normal"} tissue response`; $("#pressDetails").textContent = `${record.press.force} force · ${record.press.duration} s press`;
  const valueClass = isEdema ? "edema-value" : "";
  $("#patientDataset").className = "patient-dataset";
  $("#patientDataset").innerHTML = `<div class="dataset-heading"><p class="eyebrow">Generated patient data</p><h2>Synthetic patient profile</h2></div><div class="lab-grid">
    <article class="lab-card ${valueClass}"><h3>Albumin</h3><p class="lab-value">${values.albumin_g_dL} g/dL</p><dl><dt>Causal link to edema</dt><dd>Direct and strong</dd><dt>Mechanism</dt><dd>Albumin provides much of plasma oncotic pressure. Low albumin reduces that pressure, contributing to fluid movement from vessels into tissue and pitting edema.</dd><dt>Expected if edema present</dt><dd>Low — this edema-mode generator keeps albumin below 3.0 g/dL.</dd></dl></article>
    <article class="lab-card"><h3>Hemoglobin (Hb)</h3><p class="lab-value">${values.hemoglobin_g_dL} g/dL</p><dl><dt>Causal link to edema</dt><dd>Indirect / coexists; does not cause it</dd><dt>Mechanism</dt><dd>Severe malnutrition can produce both low Hb and low albumin because they share nutritional and illness-related causes; low Hb itself does not create the oncotic-pressure mechanism.</dd><dt>Expected if edema present</dt><dd>Often also low, but not mechanistically required — edema can occur with normal Hb.</dd></dl></article>
    <article class="lab-card"><h3>Blood glucose</h3><p class="lab-value">${values.blood_glucose_mg_dL} mg/dL</p><dl><dt>Causal link to edema</dt><dd>Indirect / coexists</dd><dt>Mechanism</dt><dd>Children with severe acute malnutrition are at risk of hypoglycaemia; low glucose and edema can co-occur in the same severe deficiency state, without a direct fluid-pressure mechanism.</dd><dt>Expected if edema present</dt><dd>May be low-normal to low; it is not a direct cause of tissue fluid retention.</dd></dl></article>
  </div>`;
}

function showCurveVerdict(payload) {
  const d = payload.deterministic;
  $("#recoveryTime").textContent = d.time_to_90_recovery === null ? `>${d.window_seconds}s` : `${d.time_to_90_recovery}s`;
  $("#deterministicVerdict").textContent = d.pit_detected ? "Pit detected" : "No pit"; $("#deterministicReason").textContent = d.reason;
}

function animate(now) {
  if (!latest) return;
  const elapsed = (now - playbackStart) / 1000, series = latest.simulation.series, total = series.at(-1).t, visualTime = Math.min(elapsed * 2.2, total), depth = depthAt(series, visualTime);
  const scanning = visualTime >= latest.simulation.duration + VISION_SCAN_DELAY && now - lastVisionScan >= VISION_SCAN_INTERVAL;
  if (scanning) { drawFabric(depth, false); scanForPit(now, visualTime); }
  else if (visualTime < latest.simulation.duration + VISION_SCAN_DELAY) scanForPit(now, visualTime);
  drawFabric(depth, true);
  $("#currentDepth").textContent = depth.toFixed(3);
  $("#releaseTime").textContent = visualTime < latest.simulation.duration ? "Press held" : `${Math.max(0, visualTime - latest.simulation.duration).toFixed(1)} s after release`;
  if (visualTime < total) animationId = requestAnimationFrame(animate);
}

async function runPatientSimulation(record) {
  cancelAnimationFrame(animationId);
  const runId = ++simulationRunId;
  latest = null;
  resetVisionDetector("Capturing unindented baseline");
  drawFabric(0);
  const request = { mode: record.label.tissue_mode, force: record.press.force, duration: record.press.duration, parameters: record.press.parameters };
  const response = await fetch("/api/simulate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request) });
  if (!response.ok) throw new Error(await response.text());
  const payload = await response.json();
  if (runId !== simulationRunId) return;
  latest = payload;
  drawGraph(latest.simulation.series, latest.simulation.duration);
  showCurveVerdict(latest);
  drawFabric(0, false);
  captureVisionBaseline();
  drawFabric(0);
  playbackStart = performance.now();
  animationId = requestAnimationFrame(animate);
}

$("#generateButton").addEventListener("click", async () => {
  const button = $("#generateButton"); button.disabled = true; button.textContent = "Generating…";
  try {
    const response = await fetch("/api/dataset/generate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ label: $("#mode").value === "edema" ? "yes" : "no", count: 1, force: +$("#force").value, duration: +$("#duration").value }) });
    if (!response.ok) throw new Error(await response.text());
    patient = (await response.json()).records.at(-1);
    showPatient(patient);
    await runPatientSimulation(patient);
  } catch (error) { alert(`Could not generate the patient: ${error.message}`); }
  finally { button.disabled = false; button.textContent = "Generate synthetic patient"; }
});

$("#mode").addEventListener("input", setModeBadge);
$("#force").addEventListener("input", updatePressControls);
$("#duration").addEventListener("input", updatePressControls);
fabric.addEventListener("click", event => {
  const box = fabric.getBoundingClientRect();
  pressPoint = { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height };
  cancelAnimationFrame(animationId); animationId = 0; simulationRunId += 1; latest = null;
  resetVisionDetector("Press point moved — looking for pit");
  $("#currentDepth").textContent = "0.000"; $("#releaseTime").textContent = "Press point moved";
  drawFabric(0);
});
window.addEventListener("resize", () => {
  resetVisionDetector("Camera frame resized — looking for pit");
  drawFabric(0);
  if (latest) drawGraph(latest.simulation.series, latest.simulation.duration);
});

setModeBadge(); updatePressControls(); resetVisionDetector(); drawFabric(0);
