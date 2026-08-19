import { useCallback, useEffect, useRef, useState } from "react";

const STAGES = [
  { id: "questionnaire", icon: "01", label: "Questionnaire", description: "Voice + touch" },
  { id: "visual", icon: "02", label: "Visual Cues", description: "Non-invasive" },
  { id: "clinical", icon: "03", label: "Clinical Results", description: "Enter or skip" },
  { id: "results", icon: "04", label: "Final Verdict", description: "Clear next step" },
];

// Ported from NewListeningLogic/stt_session.py. The Python threshold is RMS
// 500 on int16 audio, equivalent to 500/32768 for browser Float32 samples.
const VAD = {
  sampleRate: 16000,
  chunkSize: 1280,
  silenceRms: 500 / 32768,
  endSilenceMs: 1000,
  noSpeechMs: 10000,
  maxTurnMs: 10000,
};

async function readApiResponse(response) {
  const contentType = response.headers.get("content-type") ?? "";
  const body = contentType.includes("application/json")
    ? await response.json()
    : await response.text();
  if (!response.ok) {
    const message = typeof body === "object" ? body.error ?? body.detail : body;
    throw new Error(message || `Request failed with status ${response.status}`);
  }
  return body;
}

function resampleLinear(samples, sourceRate, targetRate) {
  if (sourceRate === targetRate) return samples;
  const result = new Float32Array(Math.max(1, Math.round(samples.length * targetRate / sourceRate)));
  const ratio = sourceRate / targetRate;
  for (let index = 0; index < result.length; index += 1) {
    const position = index * ratio;
    const left = Math.floor(position);
    const right = Math.min(left + 1, samples.length - 1);
    const fraction = position - left;
    result[index] = samples[left] * (1 - fraction) + samples[right] * fraction;
  }
  return result;
}

function encodePcmWav(samples, sampleRate = VAD.sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const write = (offset, value) => [...value].forEach((character, index) =>
    view.setUint8(offset + index, character.charCodeAt(0)));
  write(0, "RIFF"); view.setUint32(4, 36 + samples.length * 2, true);
  write(8, "WAVE"); write(12, "fmt "); view.setUint32(16, 16, true);
  view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  write(36, "data"); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const clamped = Math.max(-1, Math.min(1, sample));
    view.setInt16(44 + index * 2, clamped < 0 ? clamped * 32768 : clamped * 32767, true);
  });
  return buffer;
}

function FlowMap({ active, enabled, onNavigate, idle = false }) {
  return (
    <nav className={`flow-map ${idle ? "flow-map-idle" : ""}`} aria-label="Screening stages">
      {STAGES.map((stage, index) => (
        <div className="flow-node-wrap" key={stage.id}>
          <button
            className={`flow-node ${active === stage.id ? "active" : ""}`}
            type="button"
            disabled={!enabled?.includes(stage.id)}
            onClick={() => onNavigate?.(stage.id)}
          >
            <span className="stage-icon">{stage.icon}</span>
            <span>{stage.label}</span>
            {idle && <small>{stage.description}</small>}
          </button>
          {index < STAGES.length - 1 && <span className="flow-arrow" aria-hidden="true">→</span>}
        </div>
      ))}
    </nav>
  );
}

function ActionBar({ onSkip, skipLabel = "Skip this section", children }) {
  return (
    <div className="action-bar">
      {onSkip && <button className="button quiet" type="button" onClick={onSkip}>{skipLabel} <kbd>Esc</kbd></button>}
      <div className="action-main">{children}</div>
    </div>
  );
}

function App() {
  const [stage, setStage] = useState("idle");
  const [session, setSession] = useState(null);
  const [plan, setPlan] = useState(null);
  const [busy, setBusy] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [listening, setListening] = useState(false);
  const [speechSeen, setSpeechSeen] = useState(false);
  const [inputMode, setInputMode] = useState("voice");
  const [textAnswer, setTextAnswer] = useState("");
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");
  const [lastAnswer, setLastAnswer] = useState(null);
  const [testIndex, setTestIndex] = useState(0);
  const [visualResults, setVisualResults] = useState([]);
  const [clinicalValues, setClinicalValues] = useState({});
  const [invalidatedFields, setInvalidatedFields] = useState([]);
  const [invalidatedSections, setInvalidatedSections] = useState([]);
  const [finalResult, setFinalResult] = useState(null);
  const [testInputs, setTestInputs] = useState({});
  const [currentTestResult, setCurrentTestResult] = useState(null);
  const [activeClinicalIndex, setActiveClinicalIndex] = useState(0);
  const [capturedImage, setCapturedImage] = useState(null);
  const [capturedImageUrl, setCapturedImageUrl] = useState(null);
  const cameraStreamRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const sessionRef = useRef(null);
  const captureRef = useRef(null);
  const audioRef = useRef(null);
  const audioUrlRef = useRef(null);
  const askedQuestionRef = useRef("");
  const stageRef = useRef(stage);
  const menuActionRef = useRef(null);
  const startMenuRef = useRef(null);
  const playQuestionRef = useRef(null);

  useEffect(() => { sessionRef.current = session; }, [session]);
  useEffect(() => { stageRef.current = stage; }, [stage]);

  const closeCamera = useCallback(() => {
    const stream = cameraStreamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      cameraStreamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
  }, []);

  const setCapturedAndPreview = useCallback((blob) => {
    if (capturedImageUrl) URL.revokeObjectURL(capturedImageUrl);
    setCapturedImage(blob);
    setCapturedImageUrl(URL.createObjectURL(blob));
  }, [capturedImageUrl]);

  const startCamera = useCallback(async () => {
    closeCamera();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      });
      cameraStreamRef.current = stream;
      requestAnimationFrame(() => {
        if (videoRef.current) videoRef.current.srcObject = stream;
      });
    } catch {
      setError("Camera unavailable. Use file upload instead.");
    }
  }, [closeCamera]);

  const capturePhoto = useCallback(() => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) {
        closeCamera();
        setCapturedAndPreview(blob);
      }
    }, "image/jpeg", 0.92);
  }, [closeCamera, setCapturedAndPreview]);

  const handleFileUpload = useCallback((event) => {
    const file = event.target.files?.[0];
    if (file && file.type.startsWith("image/")) {
      closeCamera();
      setCapturedAndPreview(file);
    }
  }, [closeCamera, setCapturedAndPreview]);

  const stopPlayback = useCallback(() => {
    audioRef.current?.pause();
    if (audioRef.current) audioRef.current.removeAttribute("src");
    if (audioUrlRef.current) URL.revokeObjectURL(audioUrlRef.current);
    audioUrlRef.current = null;
    setSpeaking(false);
  }, []);

  const unlockAudio = useCallback(() => {
    // Keep one media element unlocked from the user's initial click/Enter.
    // Reusing it avoids Chromium kiosk autoplay blocking later async TTS WAVs.
    const audio = audioRef.current ?? new Audio();
    audioRef.current = audio;
    const silentUrl = URL.createObjectURL(new Blob([
      encodePcmWav(new Float32Array(1600)),
    ], { type: "audio/wav" }));
    audioUrlRef.current = silentUrl;
    audio.muted = true;
    audio.src = silentUrl;
    void audio.play().then(() => { audio.muted = false; }).catch(() => {
      audio.muted = false;
    });
  }, []);

  const speakText = useCallback(async (text) => {
    try {
      const response = await fetch("/api/flows/tts/speak", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (!response.ok || (response.headers.get("content-type") ?? "").includes("application/json")) return;
      stopPlayback();
      setSpeaking(true);
      const url = URL.createObjectURL(await response.blob());
      audioUrlRef.current = url;
      const audio = audioRef.current ?? new Audio();
      audioRef.current = audio; audio.muted = false; audio.src = url;
      await new Promise((resolve, reject) => {
        audio.onended = resolve;
        audio.onerror = reject;
        audio.play().catch(reject);
      });
    } catch { /* The visible physical-input message remains the fallback. */ }
    finally { setSpeaking(false); }
  }, [stopPlayback]);

  const closeCapture = useCallback(() => {
    const capture = captureRef.current;
    if (!capture) return null;
    capture.active = false;
    if (capture.timer) window.clearTimeout(capture.timer);
    try { capture.processor?.disconnect(); } catch { /* already closed */ }
    try { capture.source?.disconnect(); } catch { /* already closed */ }
    capture.stream?.getTracks().forEach((track) => track.stop());
    capture.context?.close().catch(() => {});
    captureRef.current = null;
    setListening(false);
    setSpeechSeen(false);
    return capture;
  }, []);

  const submitAnswer = useCallback(async (body, questionId, contentType) => {
    const current = sessionRef.current;
    if (!current?.session_id) return;
    setBusy(true); setError(""); setNotice("");
    try {
      const response = await fetch(
        `/api/flows/questionnaire/session/${current.session_id}/answer`,
        { method: "POST", headers: { "Content-Type": contentType, "X-Question-Id": questionId }, body },
      );
      const next = await readApiResponse(response);
      if (next.status === "repeat") {
        if (next.last_match?.value === "repeat") {
          setSession(next);
          setInputMode("voice");
          askedQuestionRef.current = next.question.id;
          setNotice("Repeating the question.");
          void playQuestionRef.current?.();
          return;
        }
        const message = "I didn't catch that. Please enter a value through the physical device.";
        setSession(next);
        setInputMode("physical");
        setNotice(message);
        void speakText(message);
      } else {
        setLastAnswer({ value: next.matched?.value, heard: next.matched?.transcription });
        setSession(next);
        setTextAnswer("");
        setInputMode("voice");
        askedQuestionRef.current = "";
        if (next.status === "complete") {
          if (next.result.invalidated) {
            setInvalidatedSections((items) => [...new Set([...items, "questionnaire"])]);
          }
          setPlan({
            visual_cues: next.result.visual_cues ?? [],
            clinical_fields: next.result.clinical_fields ?? [],
            dominant_category: next.result.dominant_category,
          });
          setStage("visual");
        }
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not submit the answer.");
    } finally { setBusy(false); }
  }, [speakText]);

  const finishListening = useCallback(async (submit = false, noSpeech = false) => {
    const capture = closeCapture();
    if (!capture) return;
    if (noSpeech) {
      const message = "I didn't catch that. Please enter a value through the physical device.";
      setInputMode("physical");
      setNotice(message);
      void speakText(message);
      return;
    }
    if (!submit || !capture.speechDetected || !capture.samples.length) return;
    const length = capture.samples.reduce((sum, samples) => sum + samples.length, 0);
    const merged = new Float32Array(length);
    let offset = 0;
    capture.samples.forEach((samples) => { merged.set(samples, offset); offset += samples.length; });
    const samples16k = resampleLinear(merged, capture.context.sampleRate, VAD.sampleRate);
    await submitAnswer(
      new Blob([encodePcmWav(samples16k)], { type: "audio/wav" }),
      capture.questionId,
      "audio/wav",
    );
  }, [closeCapture, speakText, submitAnswer]);

  const startListening = useCallback(async () => {
    const current = sessionRef.current;
    if (!current?.question || current.audio?.stt_provider === "mock") {
      setInputMode("physical");
      setNotice(current?.audio?.stt_provider === "mock" ? "Mock STT is active — use a physical answer." : "");
      return;
    }
    closeCapture();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContextClass();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(2048, 1, 1);
      const capture = {
        active: true, context, stream, source, processor, samples: [], remainder: new Float32Array(0),
        speechDetected: false, silenceStarted: null, startedAt: performance.now(),
        questionId: current.question.id, timer: null,
      };
      captureRef.current = capture;
      processor.onaudioprocess = (event) => {
        if (!capture.active) return;
        const input = event.inputBuffer.getChannelData(0);
        const joined = new Float32Array(capture.remainder.length + input.length);
        joined.set(capture.remainder); joined.set(input, capture.remainder.length);
        let position = 0;
        while (position + VAD.chunkSize <= joined.length) {
          const chunk = joined.slice(position, position + VAD.chunkSize);
          capture.samples.push(chunk);
          const rms = Math.sqrt(chunk.reduce((sum, sample) => sum + sample * sample, 0) / chunk.length);
          const now = performance.now();
          if (rms >= VAD.silenceRms) {
            capture.speechDetected = true; capture.silenceStarted = null; setSpeechSeen(true);
          } else {
            capture.silenceStarted ??= now;
            const silenceMs = now - capture.silenceStarted;
            if (capture.speechDetected && silenceMs >= VAD.endSilenceMs) {
              void finishListening(true, false); return;
            }
            if (!capture.speechDetected && silenceMs >= VAD.noSpeechMs) {
              void finishListening(false, true); return;
            }
          }
          position += VAD.chunkSize;
        }
        capture.remainder = joined.slice(position);
      };
      source.connect(processor); processor.connect(context.destination);
      capture.timer = window.setTimeout(() => {
        void finishListening(capture.speechDetected, !capture.speechDetected);
      }, VAD.maxTurnMs);
      setInputMode("voice"); setListening(true);
      setNotice((message) => message.startsWith("I couldn't") ? message : "");
    } catch {
      closeCapture(); setInputMode("physical");
      setNotice("Microphone unavailable — use the physical input device.");
    }
  }, [closeCapture, finishListening]);

  const finishMenuListening = useCallback(async (submit = false) => {
    const capture = closeCapture();
    if (!capture) return;
    if (submit && capture.speechDetected && capture.samples.length) {
      const length = capture.samples.reduce((sum, samples) => sum + samples.length, 0);
      const merged = new Float32Array(length);
      let offset = 0;
      capture.samples.forEach((samples) => { merged.set(samples, offset); offset += samples.length; });
      const samples16k = resampleLinear(merged, capture.context.sampleRate, VAD.sampleRate);
      try {
        const response = await fetch("/api/flows/stt/transcribe", {
          method: "POST", headers: { "Content-Type": "audio/wav" },
          body: new Blob([encodePcmWav(samples16k)], { type: "audio/wav" }),
        });
        const transcript = await readApiResponse(response);
        const handled = await menuActionRef.current?.(transcript);
        if (!handled) await speakText("Command not recognized. Say continue, skip section, next, or go home.");
      } catch { /* Keyboard control remains available. */ }
    }
    window.setTimeout(() => startMenuRef.current?.(), 300);
  }, [closeCapture, speakText]);

  const startMenuListening = useCallback(async () => {
    if (!["visual", "tests", "clinical", "results"].includes(stageRef.current) || captureRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
      });
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      const context = new AudioContextClass();
      const source = context.createMediaStreamSource(stream);
      const processor = context.createScriptProcessor(2048, 1, 1);
      const capture = {
        active: true, context, stream, source, processor, samples: [], remainder: new Float32Array(0),
        speechDetected: false, silenceStarted: null, timer: null,
      };
      captureRef.current = capture;
      processor.onaudioprocess = (event) => {
        if (!capture.active) return;
        const input = event.inputBuffer.getChannelData(0);
        const joined = new Float32Array(capture.remainder.length + input.length);
        joined.set(capture.remainder); joined.set(input, capture.remainder.length);
        let position = 0;
        while (position + VAD.chunkSize <= joined.length) {
          const chunk = joined.slice(position, position + VAD.chunkSize);
          capture.samples.push(chunk);
          const rms = Math.sqrt(chunk.reduce((sum, sample) => sum + sample * sample, 0) / chunk.length);
          const now = performance.now();
          if (rms >= VAD.silenceRms) {
            capture.speechDetected = true; capture.silenceStarted = null; setSpeechSeen(true);
          } else {
            capture.silenceStarted ??= now;
            if (capture.speechDetected && now - capture.silenceStarted >= VAD.endSilenceMs) {
              void finishMenuListening(true); return;
            }
          }
          position += VAD.chunkSize;
        }
        capture.remainder = joined.slice(position);
      };
      source.connect(processor); processor.connect(context.destination);
      capture.timer = window.setTimeout(() => void finishMenuListening(capture.speechDetected), VAD.maxTurnMs);
      setListening(true); setSpeechSeen(false);
    } catch { /* Keyboard-only control stays functional when the mic is unavailable. */ }
  }, [finishMenuListening]);

  useEffect(() => { startMenuRef.current = startMenuListening; }, [startMenuListening]);

  const playQuestion = useCallback(async () => {
    const current = sessionRef.current;
    if (!current?.question || !current.session_id) return;
    closeCapture(); stopPlayback(); setSpeaking(true);
    try {
      const response = await fetch(
        `/api/flows/questionnaire/session/${current.session_id}/ask`, { method: "POST" },
      );
      if (!response.ok) await readApiResponse(response);
      else if ((response.headers.get("content-type") ?? "").includes("application/json")) {
        await response.json();
      } else {
        const audioUrl = URL.createObjectURL(await response.blob());
        audioUrlRef.current = audioUrl;
        await new Promise((resolve, reject) => {
          const audio = audioRef.current ?? new Audio(); audioRef.current = audio;
          audio.muted = false; audio.src = audioUrl;
          audio.onended = resolve; audio.onerror = reject; audio.play().catch(reject);
        });
      }
    } catch (caught) {
      setError(caught?.name === "NotAllowedError"
        ? "Audio was blocked by the browser. Press Repeat question once to enable it."
        : `Question audio unavailable: ${caught instanceof Error ? caught.message : "speaker playback failed"}`);
    } finally {
      stopPlayback();
      await startListening();
    }
  }, [closeCapture, startListening, stopPlayback]);

  useEffect(() => { playQuestionRef.current = playQuestion; }, [playQuestion]);

  useEffect(() => {
    const questionId = stage === "questionnaire" ? session?.question?.id : null;
    if (!questionId || busy || askedQuestionRef.current === questionId) return;
    askedQuestionRef.current = questionId;
    void playQuestion();
  }, [busy, playQuestion, session?.question?.id, stage]);

  const usePhysicalInput = useCallback(() => {
    if (stage !== "questionnaire") return;
    closeCapture(); setInputMode("physical"); setNotice("Physical input active for this question.");
  }, [closeCapture, stage]);

  useEffect(() => () => {
    closeCapture(); stopPlayback(); closeCamera();
    if (capturedImageUrl) URL.revokeObjectURL(capturedImageUrl);
  }, [closeCapture, stopPlayback, closeCamera, capturedImageUrl]);

  const resetToIdle = () => {
    closeCapture(); stopPlayback(); closeCamera();
    if (capturedImageUrl) { URL.revokeObjectURL(capturedImageUrl); setCapturedImageUrl(null); }
    setCapturedImage(null);
    setStage("idle"); setSession(null); setPlan(null);
    setFinalResult(null); setVisualResults([]); setClinicalValues({}); setInvalidatedFields([]);
    setInvalidatedSections([]); setTestInputs({}); setNotice(""); setError(""); setLastAnswer(null);
    setCurrentTestResult(null); setActiveClinicalIndex(0);
  };

  const startSession = async () => {
    unlockAudio();
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/flows/questionnaire/session", { method: "POST" });
      const started = await readApiResponse(response);
      setSession(started); sessionRef.current = started; setStage("questionnaire");
      askedQuestionRef.current = ""; setInputMode("voice");
    } catch { setError("Could not start screening."); }
    finally { setBusy(false); }
  };

  const retryLast = async () => {
    if (!session?.session_id) return;
    closeCapture(); setBusy(true);
    try {
      const response = await fetch(`/api/flows/questionnaire/session/${session.session_id}/retry`, { method: "POST" });
      const retried = await readApiResponse(response);
      setSession(retried); setLastAnswer(null); askedQuestionRef.current = ""; setInputMode("voice");
    } catch { setError("Could not reopen the previous answer."); }
    finally { setBusy(false); }
  };

  const skipQuestionnaire = async () => {
    closeCapture(); setBusy(true); setError("");
    try {
      const response = await fetch(`/api/flows/questionnaire/session/${session.session_id}/skip`, { method: "POST" });
      const skipped = await readApiResponse(response);
      setSession(skipped); setInvalidatedSections((items) => [...new Set([...items, "questionnaire"])]);
      setPlan({
        visual_cues: skipped.result.visual_cues,
        clinical_fields: skipped.result.clinical_fields,
        dominant_category: skipped.result.dominant_category,
      });
      setStage("visual");
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not skip questionnaire."); }
    finally { setBusy(false); }
  };

  const sendPhysical = (value) => {
    if (!session?.question || busy) return;
    usePhysicalInput();
    void submitAnswer(
      JSON.stringify({ text: String(value), question_id: session.question.id }),
      session.question.id,
      "application/json",
    );
  };

  const skipVisualSection = () => {
    setInvalidatedSections((items) => [...new Set([...items, "visual_cues"])]);
    setStage("clinical");
  };

  const beginTests = () => {
    const ageMonths = session?.result?.answers?.age_months
      ?? (Number(session?.result?.answers?.age_years) * 12 || "");
    setTestInputs((values) => ({
      ...values,
      weight_height_z: {
        ...(ageMonths !== "" ? { age_months: String(ageMonths) } : {}),
        ...(values.weight_height_z ?? {}),
      },
    }));
    setTestIndex(0);
    setCurrentTestResult(null);
    setStage("tests");
  };

  const moveToNextTest = () => {
    setCurrentTestResult(null);
    closeCamera();
    if (capturedImageUrl) { URL.revokeObjectURL(capturedImageUrl); setCapturedImageUrl(null); }
    setCapturedImage(null);
    if (testIndex + 1 < plan.visual_cues.length) setTestIndex((index) => index + 1);
    else { setActiveClinicalIndex(0); setStage("clinical"); }
  };

  // Attach camera stream to video element when it appears
  useEffect(() => {
    const stream = cameraStreamRef.current;
    const video = videoRef.current;
    if (stream && video && !video.srcObject) {
      video.srcObject = stream;
    }
  });

  const submitVisionTest = useCallback(async (test, imageBlob) => {
    setBusy(true); setError("");
    try {
      const response = await fetch(
        `/api/flows/screening/vision/${test.id}?population=${session.result.population}`,
        {
          method: "POST",
          headers: { "Content-Type": "image/jpeg" },
          body: imageBlob,
        },
      );
      const result = await readApiResponse(response);
      const displayResult = {
        classification: result.classification,
        display: `Scores: ${Object.entries(result.scores).map(([k, v]) => `${k}: ${(v * 100).toFixed(1)}%`).join(", ")}`,
        threshold: `Confidence: ${(result.confidence * 100).toFixed(1)}%`,
        spoken_text: `${test.name}. Classification: ${result.classification}. Confidence: ${(result.confidence * 100).toFixed(1)} percent.`,
        vision_scores: result.scores,
        vision_primary: result.primary_score,
      };
      // Store the vision result in visualResults for final fusion
      setVisualResults((items) => [
        ...items.filter((item) => item.test_id !== test.id),
        {
          test_id: test.id,
          value: { scores: result.scores },
          confidence: result.primary_score,
          status: "complete",
        },
      ]);
      setCurrentTestResult(displayResult);
      await speakText(displayResult.spoken_text);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not classify image.");
    } finally { setBusy(false); }
  }, [session, speakText]);

  const advanceTest = async (skip = false) => {
    closeCapture();
    const test = plan.visual_cues[testIndex];
    if (currentTestResult && !skip) {
      moveToNextTest();
      return;
    }
    const physical = test.execution_type === "physical";
    const camera = !physical && test.availability.available;
    const physicalValue = testInputs[test.id] ?? {};
    if (!skip && physical) {
      const missing = (test.input_fields ?? []).find(
        (field) => physicalValue[field.id] === undefined || physicalValue[field.id] === "",
      );
      if (missing) {
        setError(`Enter ${missing.label.toLowerCase()} before continuing.`);
        return;
      }
      const invalid = (test.input_fields ?? []).find((field) => {
        if (field.type !== "number") return false;
        const value = Number(physicalValue[field.id]);
        return !Number.isFinite(value)
          || (field.min !== undefined && value < field.min)
          || (field.max !== undefined && value > field.max);
      });
      if (invalid) {
        setError(`Enter ${invalid.label.toLowerCase()} between ${invalid.min} and ${invalid.max}.`);
        return;
      }
    }
    if (!skip && camera && !capturedImage) {
      setError("Capture or upload an image before continuing.");
      return;
    }
    setError("");
    const recorded = {
      test_id: test.id,
      value: physical && !skip ? physicalValue : null,
      invalidated: skip,
      status: skip ? "skipped" : physical ? "complete" : camera ? "complete" : "no_model_installed",
    };
    setVisualResults((items) => [...items.filter((item) => item.test_id !== test.id), recorded]);
    if (skip) {
      closeCamera();
      if (capturedImageUrl) { URL.revokeObjectURL(capturedImageUrl); setCapturedImageUrl(null); }
      setCapturedImage(null);
      moveToNextTest();
      return;
    }
    if (physical) {
      setBusy(true);
      try {
        const response = await fetch("/api/flows/screening/test-result", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            population: session.result.population,
            test_id: test.id,
            value: physicalValue,
          }),
        });
        const result = await readApiResponse(response);
        setCurrentTestResult(result);
        await speakText(result.spoken_text);
      } catch (caught) {
        setError(caught instanceof Error ? caught.message : "Could not classify measurement.");
      } finally { setBusy(false); }
      return;
    }
    if (camera && capturedImage) {
      await submitVisionTest(test, capturedImage);
      return;
    }
    // No model installed fallback
    const result = {
      classification: "No model installed",
      display: "This test cannot produce a result yet.",
      spoken_text: `${test.name}. No model is installed, so no classification was produced.`,
    };
    setCurrentTestResult(result);
    await speakText(result.spoken_text);
  };

  const calculateFinal = async (sections = invalidatedSections) => {
    if (!session?.result) return;
    setBusy(true); setError("");
    try {
      const response = await fetch("/api/flows/screening/result", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          population: session.result.population,
          answers: session.result.answers,
          scores: session.result.scores,
          invalidated_sections: sections,
          visual_results: visualResults,
          clinical_values: clinicalValues,
          invalidated_fields: invalidatedFields,
        }),
      });
      const result = await readApiResponse(response);
      setFinalResult(result); setStage("results");
      await speakText(result.spoken_text);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Could not calculate result."); }
    finally { setBusy(false); }
  };

  const skipClinical = () => {
    const sections = [...new Set([...invalidatedSections, "clinical_results"])];
    setInvalidatedSections(sections);
    void calculateFinal(sections);
  };

  const enabledStages = session?.result
    ? ["questionnaire", "visual", "clinical", "results"]
    : session ? ["questionnaire"] : [];
  const mapStage = stage === "tests" ? "visual" : stage;
  const navigate = (target) => {
    if (target === "results") void calculateFinal();
    else { closeCapture(); setStage(target); }
  };

  useEffect(() => {
    if (stage !== "tests" || currentTestResult) return;
    const frame = window.requestAnimationFrame(() => {
      document.querySelector("[data-test-field]")?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [currentTestResult, stage, testIndex]);

  useEffect(() => {
    if (stage !== "clinical") return;
    const frame = window.requestAnimationFrame(() => {
      document.querySelector(`[data-clinical-index="${activeClinicalIndex}"]`)?.focus();
    });
    return () => window.cancelAnimationFrame(frame);
  }, [activeClinicalIndex, stage]);

  useEffect(() => {
    menuActionRef.current = async (transcript) => {
      const text = String(transcript?.text ?? "").toLowerCase().trim();
      const has = (...phrases) => phrases.some((phrase) => text.includes(phrase));
      if (has("go home", "return home", "finish screening")) {
        resetToIdle();
        return true;
      }
      if (has("questionnaire") && session?.result) { navigate("questionnaire"); return true; }
      if (has("visual cues", "visual section") && session?.result) { navigate("visual"); return true; }
      if (has("clinical reports", "clinical section") && session?.result) { navigate("clinical"); return true; }
      if (has("final verdict", "show result", "show results") && session?.result) { await calculateFinal(); return true; }

      if (has("skip section")) {
        if (stage === "visual" || stage === "tests") skipVisualSection();
        else if (stage === "clinical") skipClinical();
        else return false;
        return true;
      }
      if (stage === "tests" && has("skip test", "skip this test")) {
        await advanceTest(true);
        return true;
      }

      if (stage === "tests") {
        const test = plan?.visual_cues?.[testIndex];
        const fields = test?.input_fields ?? [];
        const activeId = document.activeElement?.dataset?.testField;
        const field = fields.find((item) => item.id === activeId)
          ?? fields.find((item) => (testInputs[test.id] ?? {})[item.id] === undefined || (testInputs[test.id] ?? {})[item.id] === "");
        let spokenValue = transcript?.number;
        if (field?.type === "select") {
          const wanted = has("female", "woman", "girl") ? "female"
            : has("male", "man", "boy") ? "male"
            : has("yes", "positive", "present") ? "yes"
            : has("no", "negative", "absent") ? "no" : null;
          spokenValue = field.options?.find((option) => option.value === wanted)?.value ?? null;
        } else if (field?.type === "text") {
          spokenValue = text;
        }
        if (field && spokenValue !== null && spokenValue !== undefined) {
          setTestInputs((all) => ({
            ...all, [test.id]: { ...(all[test.id] ?? {}), [field.id]: String(spokenValue) },
          }));
          const fieldIndex = fields.findIndex((item) => item.id === field.id);
          window.setTimeout(() => document.querySelector(
            `[data-test-field="${fields[fieldIndex + 1]?.id ?? field.id}"]`,
          )?.focus(), 0);
          await speakText(`${field.label} set to ${spokenValue}.`);
          return true;
        }
      }

      if (stage === "clinical") {
        const field = plan?.clinical_fields?.[activeClinicalIndex];
        const spokenValue = field?.type === "text" ? text : transcript?.number;
        if (field && spokenValue !== null && spokenValue !== undefined && spokenValue !== "") {
          setClinicalValues((values) => ({ ...values, [field.id]: String(spokenValue) }));
          await speakText(`${field.label} set to ${spokenValue}.`);
          if (activeClinicalIndex + 1 < plan.clinical_fields.length) {
            setActiveClinicalIndex((index) => index + 1);
          }
          return true;
        }
      }

      if (stage === "tests" && has("open camera", "start camera", "take photo")) {
        startCamera();
        return true;
      }
      if (stage === "tests" && has("capture", "take picture", "snap")) {
        if (cameraStreamRef.current) { capturePhoto(); return true; }
      }
      if (stage === "tests" && has("retake", "try again")) {
        if (capturedImage) {
          if (capturedImageUrl) { URL.revokeObjectURL(capturedImageUrl); setCapturedImageUrl(null); }
          setCapturedImage(null);
          return true;
        }
      }

      if (has("continue", "next", "begin tests", "start tests", "save measurement", "classify")) {
        if (stage === "visual") beginTests();
        else if (stage === "tests") await advanceTest(false);
        else if (stage === "clinical") await calculateFinal();
        else if (stage === "results") resetToIdle();
        else return false;
        return true;
      }
      return false;
    };
  });

  useEffect(() => {
    if (!["visual", "tests", "clinical", "results"].includes(stage) || speaking || busy) return;
    const timer = window.setTimeout(() => startMenuListening(), 350);
    return () => { window.clearTimeout(timer); closeCapture(); };
  }, [busy, closeCapture, speaking, stage, startMenuListening, testIndex]);

  useEffect(() => {
    const onKeyDown = (event) => {
      if (event.ctrlKey || event.altKey || event.metaKey) return;
      const isInput = ["INPUT", "TEXTAREA", "SELECT"].includes(event.target?.tagName);

      if (listening && stage !== "questionnaire") closeCapture();

      if (stage === "idle" && event.key === "Enter") {
        event.preventDefault();
        if (!busy) void startSession();
        return;
      }
      if (stage === "questionnaire") {
        if (listening) usePhysicalInput();
        const question = session?.question;
        if (event.key === "Escape" && session?.population) {
          event.preventDefault();
          void skipQuestionnaire();
          return;
        }
        if (!question && session?.result && event.key === "Enter") {
          event.preventDefault(); setStage("visual"); return;
        }
        if (question?.type === "select" && ["[", "]"].includes(event.key)) {
          const wanted = event.key === "[" ? "no" : "yes";
          const option = question.options?.find(
            (item) => String(item.value).toLowerCase() === wanted,
          );
          if (option) {
            event.preventDefault();
            usePhysicalInput();
            sendPhysical(option.value);
          }
        } else if (question?.type === "select" && /^[1-9]$/.test(event.key)) {
          const option = question.options?.[Number(event.key) - 1];
          if (option) {
            event.preventDefault();
            usePhysicalInput();
            sendPhysical(option.value);
          }
        } else if (question?.type === "number" && !isInput) {
          if (/^[0-9.]$/.test(event.key)) {
            event.preventDefault();
            usePhysicalInput();
            setTextAnswer((value) => `${value}${event.key}`);
          } else if (event.key === "Backspace") {
            event.preventDefault();
            usePhysicalInput();
            setTextAnswer((value) => value.slice(0, -1));
          } else if (event.key === "Enter" && textAnswer !== "") {
            event.preventDefault(); sendPhysical(textAnswer);
          }
        }
        return;
      }
      if (stage === "tests" && !currentTestResult && !isInput) {
        const test = plan?.visual_cues?.[testIndex];
        const camera = test && !test.execution_type && test.availability.available;
        if (camera && !capturedImage && event.key === "c") {
          event.preventDefault(); startCamera(); return;
        }
        if (camera && !capturedImage && event.key === "u") {
          event.preventDefault(); document.getElementById("vision-file-input")?.click(); return;
        }
        if (camera && cameraStreamRef.current && event.code === "Space") {
          event.preventDefault(); capturePhoto(); return;
        }
        if (camera && capturedImage && event.key === "r") {
          event.preventDefault();
          if (capturedImageUrl) { URL.revokeObjectURL(capturedImageUrl); setCapturedImageUrl(null); }
          setCapturedImage(null);
          return;
        }
      }
      if (stage === "tests" && event.key === "Enter") {
        event.preventDefault();
        if (currentTestResult) {
          void advanceTest(false);
          return;
        }
        const fields = plan?.visual_cues?.[testIndex]?.input_fields ?? [];
        const fieldId = event.target?.dataset?.testField;
        const fieldIndex = fields.findIndex((field) => field.id === fieldId);
        if (fieldIndex >= 0 && fieldIndex + 1 < fields.length) {
          document.querySelector(`[data-test-field="${fields[fieldIndex + 1].id}"]`)?.focus();
        } else {
          void advanceTest(false);
        }
        return;
      }
      if (stage === "clinical" && event.key === "Enter") {
        event.preventDefault();
        const fields = plan?.clinical_fields ?? [];
        let nextIndex = activeClinicalIndex + 1;
        while (nextIndex < fields.length && invalidatedFields.includes(fields[nextIndex].id)) nextIndex += 1;
        if (nextIndex < fields.length) setActiveClinicalIndex(nextIndex);
        else void calculateFinal();
        return;
      }
      if (event.key === "Escape") {
        event.preventDefault();
        if (stage === "visual") skipVisualSection();
        else if (stage === "tests") void advanceTest(true);
        else if (stage === "clinical") skipClinical();
        return;
      }
      if (isInput) return;
      if (event.key !== "Enter" || busy) return;
      event.preventDefault();
      if (stage === "visual") beginTests();
      else if (stage === "tests") void advanceTest(false);
      else if (stage === "clinical") void calculateFinal();
      else if (stage === "results") resetToIdle();
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  });

  const renderIdle = () => (
    <section className="idle-screen">
      <div className="idle-copy">
        <p className="eyebrow">Non invasive Screening</p>
        <h2>A clear path from questions to a practical next step.</h2>
        <p>Voice-led anemia and malnutrition risk screening that completes every stage unless you skip it.</p>
      </div>
      <FlowMap idle active={null} enabled={[]} />
      <button className="button primary start-button" type="button" onClick={startSession} disabled={busy}>
        {busy ? "Preparing…" : "Start screening"} <span>→</span>{!busy && <kbd>Enter</kbd>}
      </button>
    </section>
  );

  const renderQuestionnaire = () => {
    const question = session?.question;
    if (!question && session?.result) return (
      <section className="content execution-screen">
        <p className="eyebrow">01 / Questionnaire complete</p>
        <div className="execution-card">
          <span className="execution-mark ready">✓</span>
          <div><h2>{session.result.population.replaceAll("_", " ")}</h2>
            <p>Questionnaire scores recorded</p>
            <small>Anemia {Math.round(session.result.scores.anemia.raw * 100)}% · Malnutrition {Math.round(session.result.scores.malnutrition.raw * 100)}%</small>
          </div>
        </div>
        <ActionBar><button className="button primary" type="button" onClick={() => setStage("visual")}>Continue to visual cues →</button></ActionBar>
      </section>
    );
    if (!question) return <div className="loading">Preparing questionnaire…</div>;
    return (
      <section className="content questionnaire-screen">
        <div className="section-heading compact-heading">
          <div><p className="eyebrow">Question {Object.keys(session.answers).length + 1}</p><h2>{question.label}</h2></div>
          <div className={`voice-state ${listening ? "live" : ""}`}>
            <span className="voice-pulse" />
            <strong>{speaking ? "Speaking" : listening ? (speechSeen ? "Speech heard" : "Listening") : "Physical input"}</strong>
            <small>{inputMode === "voice" ? "Auto-submit after silence" : "Mic paused for this question"}</small>
          </div>
        </div>
        <div className="physical-input">
          {question.type === "select" ? (
            <div className="option-grid">
              {question.options.map((option, index) => (
                <button className="answer-option" type="button" key={option.value} onClick={() => sendPhysical(option.value)} disabled={busy}>
                  {String(option.value).toLowerCase() === "no" && <kbd>[</kbd>}
                  {!(["no", "yes"].includes(String(option.value).toLowerCase())) && <kbd>{index + 1}</kbd>}
                  <span>{option.label}</span>
                  {String(option.value).toLowerCase() === "yes" && <kbd>]</kbd>}
                </button>
              ))}
            </div>
          ) : (
            <form className="number-form" onSubmit={(event) => { event.preventDefault(); if (textAnswer !== "") sendPhysical(textAnswer); }}>
              <label htmlFor="number-answer">Enter number</label>
              <input id="number-answer" type="number" min={question.min} max={question.max} value={textAnswer}
                onFocus={usePhysicalInput} onChange={(event) => { usePhysicalInput(); setTextAnswer(event.target.value); }} />
              <button className="button primary" disabled={busy || textAnswer === ""}>Use answer</button>
            </form>
          )}
        </div>
        <div className="question-meta">
          <span>{notice || "You can speak, tap an option, or type a value."}</span>
          {lastAnswer && <button className="text-action" type="button" onClick={retryLast}>Undo “{String(lastAnswer.value)}”</button>}
        </div>
        <ActionBar onSkip={skipQuestionnaire}>
          <button className="button quiet" type="button" disabled={inputMode === "physical"} onClick={() => {
            askedQuestionRef.current = session.question.id;
            setInputMode("voice");
            setNotice("");
            void playQuestion();
          }}>{inputMode === "physical" ? "Voice disabled for this question" : "Repeat question"}</button>
        </ActionBar>
      </section>
    );
  };

  const renderVisual = () => (
    <section className="content visual-screen">
      <div className="section-heading inline-heading">
        <div><p className="eyebrow">02 / Non-invasive screening</p><h2>Visual Cues</h2></div>
        <p><b>{plan?.dominant_category ?? "risk"}</b> risk leads this order. Say “begin tests” or “skip section”.</p>
      </div>
      <div className="test-list">
        {(plan?.visual_cues ?? []).map((test) => (
          <article className="test-row" key={test.id}>
            <span className="test-order">{String(test.order).padStart(2, "0")}</span>
            <div><h3>{test.name}</h3><p>{test.description}</p></div>
            <span className={`availability ${test.execution_type === "physical" ? "physical" : test.availability.available ? "available" : "missing"}`}>
              {test.execution_type === "physical" ? "Physical input" : test.availability.available ? "Will run" : "No model — will be skipped"}
            </span>
          </article>
        ))}
      </div>
      <ActionBar onSkip={skipVisualSection}>
        <button className="button primary" type="button" onClick={beginTests} disabled={!plan?.visual_cues?.length}>Begin tests →</button>
      </ActionBar>
    </section>
  );

  const renderTest = () => {
    const test = plan?.visual_cues?.[testIndex];
    if (!test) return null;
    const available = test.availability.available;
    const physical = test.execution_type === "physical";
    const camera = !physical && available;
    const values = testInputs[test.id] ?? {};
    const muacValue = Number(values.muac_cm);
    const muacDomain = session.result.population === "child_under5" ? [8, 18]
      : session.result.population === "pregnant_woman" ? [15, 35] : [5, 60];
    const muacPosition = Number.isFinite(muacValue)
      ? Math.max(0, Math.min(100, ((muacValue - muacDomain[0]) / (muacDomain[1] - muacDomain[0])) * 100))
      : 0;
    return (
      <section className="content execution-screen">
        <p className="eyebrow">Test {testIndex + 1} of {plan.visual_cues.length} · {test.category} · Enter: next/classify · Esc: skip</p>
        <div className={`execution-card ${physical ? "physical-execution" : camera ? "camera-execution" : ""}`}>
          <span className={`execution-mark ${physical ? "" : available ? "ready" : "missing"}`}>{physical ? "⌨" : available ? "◉" : "—"}</span>
          <div><h2>{test.name}</h2><p>{physical ? "Enter the measured values" : available ? "Capture or upload an image" : "No model installed — test skipped"}</p><small>{test.description}</small></div>
          {physical && !currentTestResult && (
            <div className="measurement-grid">
              {(test.input_fields ?? []).map((field) => (
                <label key={field.id}>
                  <span>{field.label}{field.unit ? ` (${field.unit})` : ""}</span>
                  {field.type === "select" ? (
                    <select data-test-field={field.id} value={values[field.id] ?? ""} onChange={(event) => setTestInputs((all) => ({
                      ...all, [test.id]: { ...(all[test.id] ?? {}), [field.id]: event.target.value },
                    }))}>
                      <option value="">Choose…</option>
                      {(field.options ?? []).map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                    </select>
                  ) : (
                    <input data-test-field={field.id} type="number" min={field.min} max={field.max} step="any" value={values[field.id] ?? ""}
                      onChange={(event) => setTestInputs((all) => ({
                      ...all, [test.id]: { ...(all[test.id] ?? {}), [field.id]: event.target.value },
                    }))} />
                  )}
                </label>
              ))}
            </div>
          )}
          {camera && !currentTestResult && (
            <div className="camera-capture">
              {cameraStreamRef.current ? (
                <div className="camera-viewfinder">
                  <video ref={videoRef} autoPlay playsInline muted className="camera-video" />
                  <canvas ref={canvasRef} style={{ display: "none" }} />
                  <div className="camera-actions">
                    <button className="button primary" type="button" onClick={capturePhoto} disabled={busy}>
                      Capture <kbd>Space</kbd>
                    </button>
                    <button className="button quiet" type="button" onClick={() => { closeCamera(); }}>
                      Cancel <kbd>Esc</kbd>
                    </button>
                  </div>
                </div>
              ) : capturedImage ? (
                <div className="captured-preview">
                  <img src={capturedImageUrl} alt="Captured" className="camera-preview-img" />
                  <div className="camera-actions">
                    <button className="button quiet" type="button" onClick={() => {
                      if (capturedImageUrl) { URL.revokeObjectURL(capturedImageUrl); setCapturedImageUrl(null); }
                      setCapturedImage(null);
                    }}>Retake <kbd>R</kbd></button>
                  </div>
                </div>
              ) : (
                <div className="camera-options">
                  <button className="button primary" type="button" onClick={startCamera} disabled={busy}>
                    Open camera <kbd>C</kbd>
                  </button>
                  <label className="button quiet" style={{ cursor: "pointer" }}>
                    Upload image <kbd>U</kbd>
                    <input id="vision-file-input" type="file" accept="image/*" capture="environment" onChange={handleFileUpload} style={{ display: "none" }} />
                  </label>
                </div>
              )}
            </div>
          )}
          {currentTestResult && (
            <div className={`instant-result ${currentTestResult.vision_scores ? "vision-result" : ""}`} aria-live="polite">
              {test.id === "muac" && (
                <div className="muac-visual" role="img" aria-label={`MUAC marker at ${muacValue} centimetres`}>
                  <div className="muac-tape"><span className="danger-zone"/><span className="warning-zone"/><span className="safe-zone"/>
                    <i style={{ left: `${muacPosition}%` }}><b>{muacValue} cm</b></i>
                  </div>
                </div>
              )}
              {test.id === "weight_height_z" && (
                <div className="z-score-visual">
                  {String(currentTestResult.display).split(" · ").map((metric) => <span key={metric}>{metric}</span>)}
                </div>
              )}
              {currentTestResult.vision_scores && (
                <div className="vision-scores">
                  {Object.entries(currentTestResult.vision_scores).map(([label, score]) => (
                    <div key={label} className="score-bar">
                      <span className="score-label">{label}</span>
                      <div className="score-track"><div className="score-fill" style={{ width: `${score * 100}%` }} /></div>
                      <span className="score-value">{(score * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              )}
              {!(["muac", "weight_height_z"].includes(test.id)) && !currentTestResult.vision_scores && (
                <div className="classification-scan" aria-hidden="true"><span/><b>Measurement classified</b></div>
              )}
              <div><strong>{currentTestResult.classification}</strong><p>{currentTestResult.display}</p><small>{currentTestResult.threshold}</small></div>
            </div>
          )}
        </div>
        <ActionBar onSkip={() => advanceTest(true)} skipLabel="Skip this test">
          <button className="button primary" type="button" onClick={() => void advanceTest(false)} disabled={busy}>
            {currentTestResult ? "Next test" : physical ? "Classify measurement" : camera ? "Run inference" : "Handle test"} →
          </button>
        </ActionBar>
      </section>
    );
  };

  const renderClinical = () => (
    <section className="content clinical-screen">
      <div className="section-heading inline-heading">
        <div><p className="eyebrow">03 / Clinical reports</p><h2>Got clinical test results? Provide us those values.</h2></div>
        <p>Every relevant field is offered. Enter moves field-by-field; say “skip section” to skip.</p>
      </div>
      {(plan?.clinical_fields?.length ?? 0) === 0 ? (
        <div className="empty-panel"><strong>No clinical fields requested.</strong><span>Both questionnaire risk bands were low.</span></div>
      ) : (
        <div className="clinical-grid">
          {plan.clinical_fields.map((field) => {
            const skipped = invalidatedFields.includes(field.id);
            return (
              <label className={`clinical-field ${skipped ? "skipped" : ""}`} key={field.id}>
                <span>{field.label}{field.if_available && <small> · if available</small>}</span>
                <div><input data-clinical-index={plan.clinical_fields.indexOf(field)} type={field.type} value={clinicalValues[field.id] ?? ""} disabled={skipped}
                  onFocus={() => setActiveClinicalIndex(plan.clinical_fields.indexOf(field))}
                  onChange={(event) => setClinicalValues((values) => ({ ...values, [field.id]: event.target.value }))}
                  placeholder={field.unit || "value"} />
                  <button type="button" title={skipped ? "Restore field" : "Skip field"} onClick={() => setInvalidatedFields((items) =>
                    skipped ? items.filter((id) => id !== field.id) : [...items, field.id])}>{skipped ? "↶" : "×"}</button>
                </div>
              </label>
            );
          })}
        </div>
      )}
      <ActionBar onSkip={skipClinical}>
        <button className="button primary" type="button" onClick={() => void calculateFinal()}>Calculate verdict →</button>
      </ActionBar>
    </section>
  );

  const renderResults = () => (
    <section className="content results-screen">
      <div className="results-heading">
        <div><p className="eyebrow">04 / Screening result</p><h2>Four clear verdicts</h2></div>
        <span>{Math.round((finalResult?.completeness?.ratio ?? 0) * 100)}% entered · say “go home” when finished</span>
      </div>
      <div className="verdict-sentences">
        {(finalResult?.section_verdicts ?? []).map((item, index) => (
          <article className={item.id === "combined" ? "combined" : ""} key={item.id}>
            <span>{String(index + 1).padStart(2, "0")}</span>
            <div><small>{item.label}</small><strong>{item.verdict}</strong><p>{item.sentence}</p></div>
          </article>
        ))}
      </div>
      <div className="results-foot"><span>{finalResult?.disclaimer}</span><button className="button primary" type="button" onClick={resetToIdle}>Finish & return home <kbd>Enter</kbd></button></div>
    </section>
  );

  return (
    <main className="device-shell">
      <header className="masthead">
        <button className="brand" type="button" onClick={resetToIdle}>
          <span className="brand-leaf" aria-hidden="true">
            <svg viewBox="0 0 32 32"><path d="M27 5C16 5 8 10 7 20c5 2 11 1 15-3 4-4 5-12 5-12Z"/><path d="M5 27c4-8 9-13 17-17"/></svg>
          </span>
          <span className="brand-copy"><b>Nourish</b><small>— Edge Screening System</small></span>
        </button>
        {stage !== "idle" && <FlowMap active={mapStage} enabled={enabledStages} onNavigate={navigate} />}
      </header>
      {error && <div className="global-alert">{error}<button type="button" onClick={() => setError("")}>×</button></div>}
      <div className="screen-body">
        {stage === "idle" && renderIdle()}
        {stage === "questionnaire" && renderQuestionnaire()}
        {stage === "visual" && renderVisual()}
        {stage === "tests" && renderTest()}
        {stage === "clinical" && renderClinical()}
        {stage === "results" && renderResults()}
      </div>
    </main>
  );
}

export default App;
