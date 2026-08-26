"""
PROJECT HYDRA - Phase 2: Speech-to-Text with Session Context
File: core/stt_session.py

Single-turn session: listens for one command after wake word,
then returns. No multi-turn concept.
"""
import sounddevice as sd
import numpy as np
import time
import os
import uuid
import queue
import threading
from datetime import datetime
from core.config import get_config_value
from core import mic_state

SAMPLE_RATE           = int(get_config_value("audio", "sample_rate", default=16000))
CHANNELS              = int(get_config_value("audio", "channels", default=1))
CHUNK_SIZE            = int(get_config_value("audio", "chunk_size", default=1280))
SILENCE_RMS_THRESHOLD = int(get_config_value("stt_session", "silence_rms_threshold", default=500))
SHORT_SILENCE_SECONDS = float(get_config_value("stt_session", "short_silence_seconds", default=1.0))
LONG_SILENCE_SECONDS  = float(get_config_value("stt_session", "long_silence_seconds", default=5.0))
MAX_TURN_SECONDS      = float(get_config_value("stt_session", "max_turn_seconds", default=30))
WHISPER_MODEL_NAME    = get_config_value("stt_session", "whisper_model", default="base.en")
# Live "partial transcript" captions use their own smaller/faster model so
# that in-flight partial work never makes the *final* transcription (the one
# that actually unblocks the response) wait behind it. See _ensure_whisper()
# / _ensure_partial_whisper() below for why this needed to be split out.
PARTIAL_WHISPER_MODEL_NAME = get_config_value(
    "stt_session", "partial_whisper_model", default="tiny.en"
)
TRANSCRIBE_LANGUAGE   = get_config_value("stt_session", "language", default="en")
TRANSCRIBE_BEAM_SIZE  = int(get_config_value("stt_session", "beam_size", default=5))
WHISPER_CPU_THREADS   = int(get_config_value("stt_session", "cpu_threads", default=2))
PARTIAL_TRANSCRIBE_INTERVAL = float(
    get_config_value("stt_session", "partial_transcribe_interval", default=1.25)
)
PARTIAL_WINDOW_SECONDS = float(
    get_config_value("stt_session", "partial_window_seconds", default=2.5)
)
PARTIAL_MIN_SECONDS = float(
    get_config_value("stt_session", "partial_min_seconds", default=0.8)
)
PARTIAL_BEAM_SIZE = int(
    get_config_value("stt_session", "partial_beam_size", default=1)
)

_whisper_model = None
_transcribe_lock = threading.Lock()

# Dedicated model + lock for live partial-transcript captions. Kept entirely
# separate from _whisper_model / _transcribe_lock above: those are shared by
# the *final* transcription that the caller is actually waiting on. When
# partials shared that lock, a partial job that happened to be mid-inference
# right as the user stopped talking would make the final transcription block
# until that partial finished — since partial windows could span several
# seconds of audio, this showed up as "waits a while before responding" on
# basically every command. Using a separate, much smaller/faster model here
# means a partial job in flight is over quickly, so it's no longer the thing
# introducing latency after speech ends.
_partial_whisper_model = None
_partial_transcribe_lock = threading.Lock()
_partial_whisper_load_failed = False

def _ensure_whisper():
    """Load WhisperModel on first use. Returns the model."""
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel
        from core.mode import is_online
        print("[stt] Loading Whisper model...")
        # A model name normally triggers a Hugging Face download.  Offline
        # mode is a hard privacy boundary, so require an already-cached/local
        # model instead of allowing that request to escape.
        kwargs = {
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": WHISPER_CPU_THREADS,
        }
        if not is_online():
            kwargs["local_files_only"] = True
        try:
            _whisper_model = WhisperModel(WHISPER_MODEL_NAME, **kwargs)
        except TypeError:
            # Compatibility with older faster-whisper releases. Its downloader
            # honours HF_HUB_OFFLINE, which is set for the router process.
            if not is_online():
                os.environ.setdefault("HF_HUB_OFFLINE", "1")
            _whisper_model = WhisperModel(
                WHISPER_MODEL_NAME,
                device="cpu",
                compute_type="int8",
                cpu_threads=WHISPER_CPU_THREADS,
            )
        print("[stt] Whisper ready.")
    return _whisper_model


def _ensure_partial_whisper():
    """
    Load the small/fast partial-caption WhisperModel on first use.
    Raises if it can't be loaded (e.g. not cached locally while offline) —
    callers should let live captions fail quietly rather than falling back
    to sharing the final model's lock, which would reintroduce the delay
    this split was meant to fix.
    """
    global _partial_whisper_model, _partial_whisper_load_failed
    if _partial_whisper_load_failed:
        raise RuntimeError("partial whisper model previously failed to load")
    if _partial_whisper_model is None:
        from faster_whisper import WhisperModel
        from core.mode import is_online
        print("[stt] Loading partial-transcript Whisper model...")
        kwargs = {
            "device": "cpu",
            "compute_type": "int8",
            "cpu_threads": WHISPER_CPU_THREADS,
        }
        if not is_online():
            kwargs["local_files_only"] = True
        try:
            try:
                _partial_whisper_model = WhisperModel(PARTIAL_WHISPER_MODEL_NAME, **kwargs)
            except TypeError:
                if not is_online():
                    os.environ.setdefault("HF_HUB_OFFLINE", "1")
                _partial_whisper_model = WhisperModel(
                    PARTIAL_WHISPER_MODEL_NAME,
                    device="cpu",
                    compute_type="int8",
                    cpu_threads=WHISPER_CPU_THREADS,
                )
        except Exception:
            _partial_whisper_load_failed = True
            print(
                f"[stt] Could not load partial-transcript model "
                f"'{PARTIAL_WHISPER_MODEL_NAME}' — live captions disabled "
                f"for this run. (Final responses are unaffected.)"
            )
            raise
        print("[stt] Partial-transcript Whisper ready.")
    return _partial_whisper_model


def compute_rms(chunk):
    return np.sqrt(np.mean(chunk.astype(np.float32) ** 2))


def transcribe(audio_int16, beam_size=None):
    """Transcribe audio with optional beam_size override.
    Uses TRANSCRIBE_BEAM_SIZE for finals if beam_size not specified.
    """
    model = _ensure_whisper()
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    # Serialize final jobs that use this model. Live partial captions have their
    # own smaller model and lock, while audio capture continues independently.
    bs = beam_size if beam_size is not None else TRANSCRIBE_BEAM_SIZE
    with _transcribe_lock:
        segments, _ = model.transcribe(
            audio_f32,
            language=TRANSCRIBE_LANGUAGE,
            beam_size=bs,
            # Every wake-word activation is a new single turn, so there is no
            # previous transcription context to carry between audio windows.
            condition_on_previous_text=False,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    # Normalize Whisper's "a.m." / "p.m." to "am" / "pm"
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = text.replace("A.M.", "am").replace("P.M.", "pm")
    return text


def transcribe_partial(audio_int16):
    """
    Transcribe a rolling audio snapshot for live-caption purposes only.
    Uses its own model and lock (see _ensure_partial_whisper) so this never
    competes with — or blocks — the final transcription.
    """
    model = _ensure_partial_whisper()
    audio_f32 = audio_int16.astype(np.float32) / 32768.0
    with _partial_transcribe_lock:
        segments, _ = model.transcribe(
            audio_f32,
            language=TRANSCRIBE_LANGUAGE,
            beam_size=PARTIAL_BEAM_SIZE,
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
    text = text.replace("a.m.", "am").replace("p.m.", "pm")
    text = text.replace("A.M.", "am").replace("P.M.", "pm")
    return text


def listen_for_command(stop_event, on_partial_transcript=None):
    """
    Records ONE command after a wake word.
    Waits for speech, then waits for short silence to finalize.
    Returns transcribed text string, or empty string if nothing heard.

    When ``on_partial_transcript`` is supplied, rolling audio snapshots are
    transcribed by one background worker. Only the most recent queued snapshot
    is retained, so expensive transcription never blocks microphone capture or
    accumulates work during a long command.
    """
    print("[stt] Listening for command...")

    # Lazy import to avoid circular dependency at module load time
    try:
        from core.tts import is_speaking
    except ImportError:
        is_speaking = lambda: False

    
    import sounddevice as sd

    _input_device = None
    if sd.default.device and sd.default.device[0] is not None:
        _input_device = sd.default.device[0]

    stream = None
    for _attempt in range(8):
        try:
            stream = sd.InputStream(
                samplerate=SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=_input_device,
            )
            break
        except sd.PortAudioError as exc:
            if "unavailable" in str(exc).lower() or exc.errno in (-9985, -9986, -9987):
                time.sleep(0.4)
            else:
                raise
    if stream is None:
        print("[stt] Microphone stream unavailable after retries.")
        return ""
    stream.start()
    mic_state.mic_opened()
    print(f"[stt] Microphone stream opened on device {_input_device}.")

    capture_closed = False

    def _close_capture_stream():
        """Release the device before CPU-bound final transcription.

        ``InputStream`` continues filling PortAudio's input buffer until it is
        explicitly stopped.  A final Whisper pass can take longer than that
        buffer, producing an input-overflow flag even though no more command
        audio is needed.  Closing here also makes the microphone available to
        the wake listener while the final transcript is being produced.
        """
        nonlocal capture_closed
        if capture_closed:
            return
        capture_closed = True
        try:
            stream.stop()
        finally:
            try:
                stream.close()
            finally:
                mic_state.mic_closed()

    partial_queue = None
    partial_worker = None
    partial_accepting = threading.Event()
    partial_callback_lock = threading.Lock()
    partial_stop = object()
    partial_stopped = False
    in_silence = False  # Track when we've entered a silence gap

    # A missing optional caption model was already reported when its first load
    # failed. Do not start a worker in every later session just to repeat
    # "previously failed to load"; final command transcription is independent.
    if on_partial_transcript and not _partial_whisper_load_failed:
        partial_queue = queue.Queue(maxsize=1)
        partial_accepting.set()

        def _partial_worker():
            while True:
                snapshot = partial_queue.get()
                if snapshot is partial_stop:
                    return
                try:
                    # Dedicated fast model/lock — never blocks the final transcribe()
                    text = transcribe_partial(snapshot)
                except Exception as exc:
                    # A missing offline model is permanent for this process.
                    # Stop queueing more partial work instead of printing the
                    # same failure for every audio chunk.
                    if _partial_whisper_load_failed:
                        partial_accepting.clear()
                    else:
                        print(f"[stt] Partial transcription skipped: {exc}")
                    continue
                # The final result may have been committed while Whisper was
                # working. Never let a late partial overwrite Thinking/Idle.
                if text:
                    # Serialise callback delivery with finalisation. Once the
                    # final transcript has claimed Thinking, no late worker can
                    # put the dashboard back into Listening.
                    with partial_callback_lock:
                        if not partial_accepting.is_set():
                            continue
                        try:
                            on_partial_transcript(text)
                        except Exception as exc:
                            print(f"[stt] Partial transcript callback failed: {exc}")

        partial_worker = threading.Thread(
            target=_partial_worker,
            daemon=True,
            name="stt-partial-transcriber",
        )
        partial_worker.start()

    def _queue_partial_snapshot(recorded_chunks):
        # Don't dispatch partials during silence - avoids overlap with finalization
        if partial_queue is None or not partial_accepting.is_set() or in_silence:
            return
        full = np.concatenate(recorded_chunks, axis=0).flatten()
        max_samples = int(PARTIAL_WINDOW_SECONDS * SAMPLE_RATE)
        snapshot = full[-max_samples:].copy() if len(full) > max_samples else full.copy()
        try:
            partial_queue.put_nowait(snapshot)
        except queue.Full:
            # Replace the one waiting job with the newest audio. A worker may
            # still be processing an older snapshot, but no work piles up.
            try:
                partial_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                partial_queue.put_nowait(snapshot)
            except queue.Full:
                pass

    def _stop_partial_worker():
        nonlocal partial_stopped
        if partial_queue is None or partial_stopped:
            return
        partial_stopped = True
        with partial_callback_lock:
            partial_accepting.clear()
        # Drop an unstarted snapshot; a currently running one is allowed to
        # finish but can no longer invoke the callback.
        while True:
            try:
                partial_queue.get_nowait()
            except queue.Empty:
                break
        partial_queue.put(partial_stop)
        # Don't join - let the worker finish asynchronously. The final
        # transcribe() will acquire the lock when the partial releases it.

    try:
        recorded_chunks = []
        speech_detected = False
        silence_start   = None
        start_time      = time.time()
        next_partial_at = time.monotonic() + PARTIAL_TRANSCRIBE_INTERVAL
        timeout_reached = False

        while not stop_event.is_set():
            try:
                chunk, _ = stream.read(CHUNK_SIZE)
            except Exception as exc:
                print(f"[stt] Stream read error: {exc}")
                return ""

            # ANC reuses the existing capture endpoint instead of opening a
            # second ALSA input stream (which this USB codec does not allow).
            mic_state.publish_audio(
                chunk[:, 0],
                SAMPLE_RATE,
                time.monotonic() - len(chunk) / SAMPLE_RATE,
            )

            # Skip audio while TTS is playing (echo prevention)
            if is_speaking():
                silence_start = None
                continue

            recorded_chunks.append(chunk)

            volume    = compute_rms(chunk)
            is_silent = volume < SILENCE_RMS_THRESHOLD

            if not is_silent:
                speech_detected = True
                in_silence = False
                silence_start   = None
            else:
                in_silence = True
                if silence_start is None:
                    silence_start = time.time()

                silence_duration = time.time() - silence_start

                # After speech detected, short silence = end of turn
                if speech_detected and silence_duration >= SHORT_SILENCE_SECONDS:
                    audio = np.concatenate(recorded_chunks, axis=0).flatten()
                    _stop_partial_worker()
                    _close_capture_stream()
                    print(
                        f"[stt] End of speech detected after "
                        f"{silence_duration:.2f}s; transcribing..."
                    )
                    transcribe_started = time.monotonic()
                    text = transcribe(audio)
                    transcribe_seconds = time.monotonic() - transcribe_started
                    if text:
                        print(
                            f'[stt] Command ({transcribe_seconds:.2f}s): "{text}"'
                        )
                    else:
                        print(
                            f"[stt] (noise, ignoring; transcription took "
                            f"{transcribe_seconds:.2f}s)"
                        )
                    return text

                # No speech at all for too long — timeout
                if not speech_detected and silence_duration >= LONG_SILENCE_SECONDS:
                    print("[stt] Timeout — no speech detected.")
                    return ""

            if (
                speech_detected
                and partial_queue is not None
                and time.monotonic() >= next_partial_at
                and len(recorded_chunks) * CHUNK_SIZE / SAMPLE_RATE >= PARTIAL_MIN_SECONDS
            ):
                _queue_partial_snapshot(recorded_chunks)
                next_partial_at = time.monotonic() + PARTIAL_TRANSCRIBE_INTERVAL

            # Absolute max session time
            if time.time() - start_time >= MAX_TURN_SECONDS:
                timeout_reached = True
                break

        if timeout_reached and recorded_chunks and speech_detected:
            audio = np.concatenate(recorded_chunks, axis=0).flatten()
            _stop_partial_worker()
            _close_capture_stream()
            print("[stt] Maximum turn length reached; transcribing...")
            transcribe_started = time.monotonic()
            text = transcribe(audio)
            transcribe_seconds = time.monotonic() - transcribe_started
            if text:
                print(f'[stt] Command ({transcribe_seconds:.2f}s): "{text}"')
            else:
                print(
                    f"[stt] (noise, ignoring; transcription took "
                    f"{transcribe_seconds:.2f}s)"
                )
            return text

    finally:
        _stop_partial_worker()
        _close_capture_stream()

    return ""
