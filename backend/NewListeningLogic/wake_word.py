import os
import queue
import threading
import datetime
import time
import numpy as np
import sounddevice as sd
import sys
import traceback
from scipy.io.wavfile import write as wav_write, read as wav_read
from openwakeword.model import Model
from core.config import get_config_value
from core.alert_sounds import sound_path
from core.audio_io import play_array, record_array
from core import mic_state

LOG_FILE             = "logs/wake_word.log"
LIVE_LOG             = "logs/wake_audio.log"
SAMPLE_RATE          = int(get_config_value("audio", "sample_rate", default=16000))
CHUNK_SIZE           = int(get_config_value("audio", "chunk_size", default=1280))
CHANNELS             = int(get_config_value("audio", "channels", default=1))
DETECTION_THRESHOLD  = float(get_config_value("wake_word", "detection_threshold", default=0.05))
WAKE_WORD_MODEL_PATH = "models/hai_dra.onnx"
RESPONSE_BEEP_ENABLED = bool(get_config_value("wake_word", "response_beep_enabled", default=True))
RESPONSE_BEEP_GAIN    = float(get_config_value("wake_word", "response_beep_gain", default=0.5))
ASSISTANT_INPUT_DEVICE = get_config_value("audio", "input_device", default=None)


def create_wake_model(model_path: str):
    """Create an OpenWakeWord model across old/new package argument names."""
    attempts = [
        {"wakeword_model_paths": [model_path]},
        {"wakeword_model_paths": [model_path], "inference_framework": "onnx"},
        {"wakeword_models": [model_path], "inference_framework": "onnx"},
        {"wakeword_models": [model_path]},
    ]
    last_error = None
    for kwargs in attempts:
        try:
            return Model(**kwargs)
        except TypeError as exc:
            last_error = exc
    raise last_error


def create_response_audio(path):
    if os.path.exists(path):
        return
    import numpy as np
    rate  = 44100
    t     = np.linspace(0, 0.4, int(rate * 0.4))
    freq  = np.where(t < 0.2, 880, 1046)
    audio = (RESPONSE_BEEP_GAIN * np.sin(2 * np.pi * freq * t) * 32767).astype(np.int16)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    wav_write(path, rate, audio)


def play_response():
    if not RESPONSE_BEEP_ENABLED:
        return
    path = sound_path("trigger") or "audio/yes.wav"
    create_response_audio(path)
    rate, audio = wav_read(path)
    # Once libmpv has opened this Pi's USB ALSA output, PortAudio cannot open
    # the same device in-process—even after mpv is paused/stopped. Reuse mpv
    # for the cue instead of producing a wall of harmless-but-real dmix errors.
    try:
        from skills.online.music import MusicPlayer

        player = MusicPlayer._instance
        if (
            player is not None
            and player.mpv is not None
            and player.play_audio_cue(path, len(audio) / float(rate))
        ):
            return
    except Exception as exc:
        print(f"[wake] mpv response cue fallback failed: {exc}")
    play_array(audio, rate)


def log(message):
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")


def listen_for_wake_word(
    on_detected,
    stop_event,
    session_active=None,
    session_alive=None,
    *,
    wake_lease=None,
    claim_detected=None,
):
    """Listen continuously and hand accepted detections to ``on_detected``.

    ``wake_lease`` and ``claim_detected`` are optional so existing diagnostics
    retain their original behavior. Production passes both: every opened wake
    stream is generation-stamped, and a candidate must atomically claim that
    same generation before acknowledgement audio or command STT can start.
    """
    os.makedirs("logs",  exist_ok=True)
    os.makedirs("audio", exist_ok=True)
    create_response_audio(sound_path("trigger") or "audio/yes.wav")

    devices = sd.query_devices()
    default_input = sd.default.device[0] if sd.default.device[0] is not None else "system default"
    print(f"[wake] Default input device index: {default_input}")
    print(f"[wake] Available input devices:")
    for i, dev in enumerate(devices):
        if dev["max_input_channels"] > 0:
            print(f"       [{i}] {dev['name']}  (channels={dev['max_input_channels']}, samplerate={dev['default_samplerate']})")
    print()

    input_device = None
    if ASSISTANT_INPUT_DEVICE not in (None, "", "default"):
        if isinstance(ASSISTANT_INPUT_DEVICE, int) or str(ASSISTANT_INPUT_DEVICE).isdigit():
            candidate = int(ASSISTANT_INPUT_DEVICE)
            if 0 <= candidate < len(devices) and devices[candidate]["max_input_channels"] > 0:
                input_device = candidate
        else:
            wanted = str(ASSISTANT_INPUT_DEVICE).lower()
            for i, dev in enumerate(devices):
                if dev["max_input_channels"] > 0 and wanted in dev["name"].lower():
                    input_device = i
                    break
    for i, dev in enumerate(devices):
        if input_device is None and dev["max_input_channels"] > 0 and "USB" in dev["name"]:
            input_device = i
            break
    if input_device is None:
        for i, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and "Sound Mapper" not in dev["name"]:
                input_device = i
                break
    if input_device is not None:
        sd.default.device = (input_device, sd.default.device[1])
        print(f"[wake] Explicitly set input device to [{input_device}] {devices[input_device]['name']}")
    else:
        print(f"[wake] WARNING: No input device found, using system default ({default_input})")
    print()

    print("[wake] Testing microphone for 1 second...")
    test_audio = record_array(
        SAMPLE_RATE,
        SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
    )
    test_rms = float(np.sqrt(np.mean(test_audio.astype(np.float64) ** 2)))
    test_peak = int(np.max(np.abs(test_audio)))
    print(f"[wake] Mic test: rms={test_rms:.1f}  peak={test_peak}  {'OK' if test_peak > 10 else 'WARNING: very low or no input!'}")
    print()

    model = create_wake_model(WAKE_WORD_MODEL_PATH)
    print(f"[wake] Model ready. Listening for 'Hey Hydra'...  (threshold={DETECTION_THRESHOLD}, beep_gain={RESPONSE_BEEP_GAIN})\n")

    # Wake-word inference can occasionally take longer than one audio block
    # on a busy Pi.  Keep only a tiny, current backlog so memory and latency
    # cannot grow until PortAudio starts reporting input overflows.
    audio_q = queue.Queue(maxsize=2)
    status_q = queue.Queue(maxsize=4)

    def _callback(indata, frames, time_info, status):
        if status:
            # Console I/O inside a PortAudio callback can itself make an
            # otherwise transient status recur.  Forward it to the consumer
            # thread for reporting instead.
            try:
                status_q.put_nowait(str(status))
            except queue.Full:
                pass
        block = indata.copy()
        try:
            audio_q.put_nowait(block)
        except queue.Full:
            # Discard stale audio, not the newest block.  Wake-word detection
            # remains live and the callback never waits on a slow consumer.
            try:
                audio_q.get_nowait()
            except queue.Empty:
                pass
            try:
                audio_q.put_nowait(block)
            except queue.Full:
                pass
        mic_state.publish_audio(
            indata[:, 0],
            SAMPLE_RATE,
            getattr(time_info, "inputBufferAdcTime", None),
        )

    _max_score_seen = 0.0
    chunk_count = 0

    os.makedirs("logs", exist_ok=True)
    live_log_fh = open(LIVE_LOG, "w", encoding="utf-8")
    live_log_fh.write("# wake_audio.log — live per-chunk metrics\n")
    live_log_fh.write(f"# threshold={DETECTION_THRESHOLD}\n")
    live_log_fh.write(f"# columns: timestamp score volume peak max_seen\n")
    live_log_fh.flush()

    def session_busy():
        command_capture_active = session_active.is_set() if session_active is not None else False
        return command_capture_active or mic_state.is_exclusive_capture_requested()

    while not stop_event.is_set():
        listener_lease = wake_lease() if callable(wake_lease) else 0
        # Playback is deliberately absent from this condition. This exact USB
        # device and stream configuration were verified with simultaneous
        # capture/playback, so wake detection stays live during all output.
        if not session_busy() and listener_lease is not None:
            stream = None
            for _attempt in range(6):
                try:
                    stream = sd.InputStream(
                        samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="int16",
                        blocksize=CHUNK_SIZE,
                        callback=_callback,
                    )
                    break
                except sd.PortAudioError:
                    time.sleep(0.2)
            if stream is None:
                time.sleep(0.3)
                continue
            mic_state.mic_opened()
            try:
                with stream:
                    detected = False
                    while (
                        not stop_event.is_set()
                        and not session_busy()
                        and (
                            not callable(wake_lease)
                            or wake_lease() == listener_lease
                        )
                    ):
                        while True:
                            try:
                                status = status_q.get_nowait()
                            except queue.Empty:
                                break
                            print(f"[wake] stream status: {status}")
                        try:
                            audio = audio_q.get(timeout=0.1)
                        except queue.Empty:
                            continue

                        audio_flat = audio.flatten()
                        prediction = model.predict(audio_flat)
                        score      = max(prediction.values()) if prediction else 0.0

                        rms  = float(np.sqrt(np.mean(audio_flat.astype(np.float64) ** 2)))

                        lufs = float()
                        peak = int(np.max(np.abs(audio_flat)))

                        chunk_count += 1
                        if score > _max_score_seen:
                            _max_score_seen = score

                        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                        live_log_fh.write(f"{ts} {score:.4f} {rms:.1f} {peak} {_max_score_seen:.4f}\n")

                        if score >= DETECTION_THRESHOLD:
                            model.reset()
                            detected = True
                            break
            finally:
                # Wake capture closes before command STT opens the same input.
                # Audio output is independent and may remain active throughout.
                mic_state.mic_closed()

            while not audio_q.empty():
                audio_q.get_nowait()

            if detected:
                if callable(claim_detected):
                    try:
                        accepted = bool(claim_detected(listener_lease))
                    except Exception as exc:
                        print(f"[wake] Detection claim failed: {exc}")
                        traceback.print_exc()
                        log(
                            "DETECTION CLAIM FAILED: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        accepted = False
                    if not accepted:
                        # Typing or another newer listener generation won the
                        # race after this audio block was captured.
                        continue

                print("\n[wake] Wake word detected.")
                log("WAKE WORD DETECTED")
                # Full-duplex capture lets the wake word interrupt an active
                # reply. Stop that reply before the acknowledgement beep and
                # before command STT starts recording.
                try:
                    from core.tts import stop_speaking
                    stop_speaking()
                except Exception:
                    pass
                # mpv uses a separate output path. Duck an active track before
                # the beep; main restores the player's configured volume when
                # the command session ends.
                player = None
                voice_ducked = False
                try:
                    from skills.online.music import MusicPlayer
                    player = MusicPlayer._instance
                    if player is not None and player.now_playing is not None:
                        voice_ducked = bool(player.begin_voice_duck())
                except Exception:
                    pass
                try:
                    play_response()
                except Exception as exc:
                    # A busy/missing output must not kill wake capture.
                    print(f"[wake] Response sound failed: {exc}")
                    log(f"RESPONSE SOUND FAILED: {exc}")
                try:
                    on_detected()
                except Exception as exc:
                    if voice_ducked and player is not None:
                        try:
                            player.end_voice_duck()
                        except Exception:
                            pass
                    print(f"[wake] Detection callback failed: {exc}")
                    traceback.print_exc()
                    log(f"DETECTION CALLBACK FAILED: {type(exc).__name__}: {exc}")

        # ── Session-active mode: no mic polling ─────────────────────────────
        # A STT session is running and already holds the InputStream open.
        # Touching the device here causes ALSA conflicts and PortAudio errors.
        else:
            if session_alive is not None and not session_alive():
                if session_active is not None:
                    session_active.clear()
            time.sleep(0.1)

    live_log_fh.close()
