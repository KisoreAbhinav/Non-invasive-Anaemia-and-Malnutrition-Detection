"""
PROJECT HYDRA
File: core/mic_state.py

Shared microphone-capture state.

The deployed USB device has been verified full-duplex, so playback never asks
capture to close. This signal remains useful for diagnostics and for ensuring
the wake-word stream and command-STT stream do not both consume the single
capture endpoint at once.
"""
import threading

_mic_active = threading.Event()
_exclusive_capture = threading.Event()
_tap_lock = threading.Lock()
_audio_taps = set()


def mic_opened():
    """Call right before opening an sd.InputStream for continuous capture."""
    _mic_active.set()


def mic_closed():
    """Call right after that sd.InputStream is stopped/closed."""
    _mic_active.clear()


def is_mic_active() -> bool:
    return _mic_active.is_set()


def request_exclusive_capture():
    """Ask other capture owners to yield until a long-running session finishes."""
    _exclusive_capture.set()


def release_exclusive_capture():
    """Allow the always-on wake listener to acquire the microphone again."""
    _exclusive_capture.clear()


def is_exclusive_capture_requested() -> bool:
    return _exclusive_capture.is_set()


def register_audio_tap(callback):
    """Subscribe to audio already captured by wake-word/STT streams.

    A tap must return quickly. It receives ``(samples, sample_rate,
    adc_time)``; samples are owned by the publisher and must be copied if the
    callback retains them. This avoids opening competing ALSA capture streams.
    """
    with _tap_lock:
        _audio_taps.add(callback)


def unregister_audio_tap(callback):
    with _tap_lock:
        _audio_taps.discard(callback)


def publish_audio(samples, sample_rate, adc_time=None):
    """Fan captured microphone audio out to lightweight local processors."""
    with _tap_lock:
        callbacks = tuple(_audio_taps)
    for callback in callbacks:
        try:
            callback(samples, sample_rate, adc_time)
        except Exception:
            # A diagnostic/ANC consumer must never break voice capture.
            continue
