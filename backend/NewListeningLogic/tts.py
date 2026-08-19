"""
PROJECT HYDRA - Text-to-Speech
File: core/tts.py

Primary:  Piper TTS (lessac-medium for English, rohan-medium for Hindi)
Fallback: espeak-ng

Language is auto-detected from the text itself (Devanagari range = Hindi),
so callers never need to specify it — skills just return strings as before.

Usage from any skill:
    from core.tts import speak
    speak("Timer set for 5 minutes.")
    speak("Apple in Hindi is \u0938\u0947\u092c.")   # auto-routes to the Hindi voice
"""

import os
import re
import sys
import gc
import importlib.util
import threading
import subprocess
import tempfile
import time
import numpy as np
import sounddevice as sd
from scipy.io.wavfile import read as wav_read, write as wav_write
from core.config import get_config_value
from core.audio_io import play_array_on_bluetooth

# ── Paths ────────────────────────────────────────────────────────────────────
PIPER_MODELS = {
    "en": {
        "model":  "models/en_US-lessac-medium.onnx",
        "config": "models/en_US-lessac-medium.onnx.json",
    },
    "hi": {
        "model":  "models/hi_IN-rohan-medium.onnx",
        "config": "models/hi_IN-rohan-medium.onnx.json",
    },
}

_DEVANAGARI = re.compile(r"[\u0900-\u097F]")
OUTPUT_GAIN = float(get_config_value("tts", "output_gain", default=1.0))
ESPEAK_RATE = int(get_config_value("tts", "espeak_rate", default=150))
PIPER_LENGTH_SCALE = float(get_config_value("tts", "piper_length_scale", default=1.0))

# ── Ducking config ────────────────────────────────────────────────────────────
DUCK_VOLUME_PERCENT = float(get_config_value("tts", "duck_volume_percent", default=25.0))

# Keep Piper's ONNX session in this process. Starting the CLI for every reply
# reloads the ~63 MB voice model and adds several seconds before audio starts on
# a Raspberry Pi. The cache is populated during main.py's normal preload stage.
_piper_voices = {}
_piper_voice_lock = threading.Lock()
_piper_synthesis_lock = threading.Lock()


def _detect_lang(text: str) -> str:
    """Hindi if the text contains any Devanagari characters, else English."""
    return "hi" if _DEVANAGARI.search(text) else "en"


def _apply_gain(audio):
    gain = max(0.0, OUTPUT_GAIN)
    if gain == 1.0:
        return audio
    scaled = np.asarray(audio, dtype=np.float32) * gain
    if np.issubdtype(audio.dtype, np.integer):
        info = np.iinfo(audio.dtype)
        scaled = np.clip(scaled, info.min, info.max).astype(audio.dtype)
    return scaled


def _blocking_play(audio, samplerate):
    gc.collect()

    if audio.dtype != np.int16:
        audio = np.clip(audio * 32767, -32768, 32767).astype(np.int16)

    # PortAudio on this Pi exposes only ALSA devices, while BlueZ audio
    # outputs exist under PipeWire. Prefer a connected Bluetooth sink before
    # considering the persistent mpv/ALSA owner or the local speaker.
    bluetooth_result = play_array_on_bluetooth(
        audio,
        samplerate,
        stop_event=_interrupt,
    )
    if bluetooth_result is not None:
        return

    # libmpv and PortAudio cannot share the deployed USB ALSA output. If music
    # has initialized mpv, send synthesized speech through that same owner.
    try:
        from skills.online.music import MusicPlayer

        player = MusicPlayer._instance
        if player is not None and player.mpv is not None:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                cue_path = tmp.name
            try:
                wav_write(cue_path, samplerate, audio)
                if player.play_audio_cue(
                    cue_path,
                    len(audio) / float(samplerate),
                    interrupt_event=_interrupt,
                ):
                    return
            finally:
                try:
                    os.remove(cue_path)
                except OSError:
                    pass
    except Exception as exc:
        print(f"[tts] mpv playback fallback unavailable: {exc}")

    for attempt in range(4):
        try:
            with sd.OutputStream(
                samplerate=samplerate,
                channels=1,
                dtype="int16",
            ) as stream:
                chunk_size = 1024
                for i in range(0, len(audio), chunk_size):
                    if _interrupt.is_set():
                        return
                    stream.write(audio[i : i + chunk_size])
                time.sleep(0.05)  # let the last buffered chunk finish playing
                return
        except sd.PortAudioError as exc:
            errno = getattr(exc, "errno", None)
            if errno in (-9985, -9986, -9987, -9988):
                time.sleep(0.25)
            else:
                raise
    raise RuntimeError("TTS playback failed after retries")


# ── Mute flag ─────────────────────────────────────────────────────────────────
# stt_session checks this before recording a chunk. When True, input is suppressed.
_speaking = threading.Event()

def is_speaking() -> bool:
    return _speaking.is_set()


# ── TTS-active status signal ──────────────────────────────────────────────────
# Retained for UI/diagnostics. The verified full-duplex wake loop deliberately
# does not pause or alter capture when this is set.
_tts_active = threading.Event()

def is_tts_active() -> bool:
    """Return True if TTS playback is currently in progress."""
    return _tts_active.is_set()


# ── Playback-interrupt handle ─────────────────────────────────────────────────
# A threading.Event that makes the explicitly owned playback loop return so
# the next speak() call can take over without touching global PortAudio state.
_interrupt = threading.Event()
_speak_lock = threading.Lock()

def stop_speaking():
    """Interrupt any in-progress TTS playback immediately."""
    _interrupt.set()


# ── Engine detection ─────────────────────────────────────────────────────────
def _piper_available(lang: str) -> bool:
    paths = PIPER_MODELS.get(lang, PIPER_MODELS["en"])
    return (
        os.path.exists(paths["model"])
        and os.path.exists(paths["config"])
        and (
            importlib.util.find_spec("piper") is not None
            or _command_exists("piper")
        )
    )

def _command_exists(cmd: str) -> bool:
    import shutil
    return shutil.which(cmd) is not None


def _run_interruptible(cmd: list[str], *, input_bytes: bytes | None = None, timeout: float | None = None):
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE if input_bytes is not None else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if input_bytes is not None and proc.stdin is not None:
        try:
            proc.stdin.write(input_bytes)
            proc.stdin.close()
        except BrokenPipeError:
            pass

    start = threading.current_thread()
    deadline = None if timeout is None else (time.time() + timeout)
    while proc.poll() is None:
        if _interrupt.is_set() or (deadline is not None and time.time() > deadline):
            proc.terminate()
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise RuntimeError("TTS interrupted.")
        time.sleep(0.05)

    stdout = proc.stdout.read() if proc.stdout else b""
    stderr = proc.stderr.read() if proc.stderr else b""
    return proc.returncode, stdout, stderr


# ── Piper synthesis ──────────────────────────────────────────────────────────
def _get_piper_voice(lang: str):
    """Return a cached in-process Piper voice, loading it once if necessary."""
    lang = lang if lang in PIPER_MODELS else "en"
    voice = _piper_voices.get(lang)
    if voice is not None:
        return voice

    paths = PIPER_MODELS.get(lang, PIPER_MODELS["en"])
    with _piper_voice_lock:
        voice = _piper_voices.get(lang)
        if voice is None:
            from piper import PiperVoice

            voice = PiperVoice.load(paths["model"], paths["config"])
            _piper_voices[lang] = voice
    return voice


def preload_piper_voice(lang: str = "en") -> bool:
    """
    Load and warm a Piper voice for low-latency replies.

    Returns False when the Python Piper runtime or voice files are unavailable;
    speak() retains the existing CLI/espeak fallback in that case.
    """
    lang = lang if lang in PIPER_MODELS else "en"
    paths = PIPER_MODELS[lang]
    if (
        not os.path.exists(paths["model"])
        or not os.path.exists(paths["config"])
        or importlib.util.find_spec("piper") is None
    ):
        return False

    try:
        from piper import SynthesisConfig

        voice = _get_piper_voice(lang)
        syn_config = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
        # Run one short inference now so ONNX kernels are initialized during
        # startup instead of when the first real reply is ready to be spoken.
        with _piper_synthesis_lock:
            list(voice.synthesize("Hello.", syn_config))
        return True
    except Exception as exc:
        _piper_voices.pop(lang, None)
        print(f"[tts] Could not preload Piper voice '{lang}': {exc}")
        return False


def _speak_piper_cli(text: str, lang: str):
    """Compatibility fallback for systems with the Piper CLI only."""
    paths = PIPER_MODELS.get(lang, PIPER_MODELS["en"])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "piper",
            "--model", paths["model"],
            "--config", paths["config"],
            "--output_file", tmp_path,
        ]
        if PIPER_LENGTH_SCALE != 1.0:
            cmd += ["--length-scale", str(PIPER_LENGTH_SCALE)]

        returncode, _stdout, stderr = _run_interruptible(
            cmd,
            input_bytes=text.encode("utf-8"),
        )
        if returncode != 0:
            raise RuntimeError(f"Piper failed: {stderr.decode()}")
        if _interrupt.is_set():
            return

        rate, audio = wav_read(tmp_path)
        _blocking_play(_apply_gain(audio), rate)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def _speak_piper(text: str, lang: str):
    """
    Synthesize with the cached Piper session, then play immediately.

    Piper inference is serialized because one ONNX session must not be driven
    concurrently by normal replies and scheduler notifications.
    """
    if importlib.util.find_spec("piper") is None:
        return _speak_piper_cli(text, lang)

    try:
        from piper import SynthesisConfig

        voice = _get_piper_voice(lang)
        syn_config = SynthesisConfig(length_scale=PIPER_LENGTH_SCALE)
        synthesis_started = time.monotonic()
        with _piper_synthesis_lock:
            chunks = list(voice.synthesize(text, syn_config))
        synthesis_seconds = time.monotonic() - synthesis_started
        if _interrupt.is_set() or not chunks:
            return

        audio = np.concatenate([chunk.audio_int16_array for chunk in chunks])
        print(f"[tts] Audio ready in {synthesis_seconds:.2f}s; playing.")
        _blocking_play(_apply_gain(audio), voice.config.sample_rate)
    except (ImportError, ModuleNotFoundError):
        _speak_piper_cli(text, lang)


# ── espeak fallback ──────────────────────────────────────────────────────────
def _speak_espeak(text: str, lang: str):
    voice = "hi" if lang == "hi" else "en-us"
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        returncode, _stdout, stderr = _run_interruptible(
            ["espeak-ng", "-v", voice, "-s", str(ESPEAK_RATE), "-w", tmp_path, text],
        )
        if returncode != 0:
            raise RuntimeError(f"espeak-ng failed: {stderr.decode()}")
        if _interrupt.is_set():
            return
        rate, audio = wav_read(tmp_path)
        _blocking_play(_apply_gain(audio), rate)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


# ── Music ducking helpers ─────────────────────────────────────────────────────
def _get_music_player():
    try:
        from skills.online.music import MusicPlayer
        # Speaking a normal online answer must not instantiate libmpv and
        # reserve the Pi's audio output when music has never been requested.
        return MusicPlayer._instance
    except Exception:
        return None

def _duck_music():
    """Temporarily free the USB output while TTS is speaking."""
    player = _get_music_player()
    if player is None:
        return
    try:
        player.begin_voice_duck(volume=DUCK_VOLUME_PERCENT)
    except Exception:
        pass

def _unduck_music():
    """Balance the active music voice section after TTS finishes."""
    player = _get_music_player()
    if player is None:
        return
    try:
        player.end_voice_duck()
    except Exception:
        pass


# ── Public API ───────────────────────────────────────────────────────────────
def speak(text: str, blocking: bool = True):
    """
    Synthesise and play speech. Language is auto-detected from the text.

    While speaking:
      - Music volume is ducked to DUCK_VOLUME_PERCENT (configurable).
      - The interrupt flag is cleared on entry so a fresh stop_speaking()
        call is required for each subsequent override.
      - Sets _tts_active for status/diagnostics; wake capture remains active.

    Args:
        text:     The string to speak.
        blocking: If True (default), this call returns only after playback
                  finishes.  If False, fires in a background thread.
    """
    if not text or not text.strip():
        return

    print(f"[tts] {text}")

    lang = _detect_lang(text)

    def _run():
        # Notifications and assistant replies can originate on different
        # threads. Serialize them so a newer call cannot clear an older call's
        # interrupt flag and accidentally revive stale playback.
        with _speak_lock:
            _interrupt.clear()
            _speaking.set()          # mute mic for stt_session
            _tts_active.set()        # status/diagnostics; capture stays active
            _duck_music()
            try:
                # Hindi: use espeak-ng directly — Piper's internal espeak bridge
                # has a UTF-8 encoding bug with Devanagari characters.
                if lang == "hi" and _command_exists("espeak-ng"):
                    _speak_espeak(text, "hi")
                elif _piper_available(lang):
                    _speak_piper(text, lang)
                elif _command_exists("espeak-ng"):
                    _speak_espeak(text, lang)
                else:
                    print(f"[tts] No TTS engine available ({lang}). Would say: {text}")
            except Exception as e:
                msg = str(e)
                if "Error querying device" in msg or "No such device" in msg:
                    print("[tts] No output audio device available.")
                else:
                    print(f"[tts] Error: {e}")
            finally:
                _unduck_music()
                _tts_active.clear()
                _speaking.clear()    # un-mute mic for stt_session

    if blocking:
        _run()
    else:
        t = threading.Thread(target=_run, daemon=True)
        t.start()
