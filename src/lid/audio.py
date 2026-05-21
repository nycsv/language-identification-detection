"""Load and validate audio for language identification.

Both models in this package expect 16 kHz, 16-bit, single-channel (mono) PCM WAV.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 16_000


def validate_wav(path: str | Path) -> Path:
    """Check that `path` is an existing 16 kHz mono WAV file.

    Returns the resolved path. Raises FileNotFoundError or ValueError otherwise.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"{path}: expected {SAMPLE_RATE} Hz, got {info.samplerate} Hz"
        )
    if info.channels != 1:
        raise ValueError(
            f"{path}: expected mono audio, got {info.channels} channels"
        )
    return path


def load_wav(path: str | Path) -> np.ndarray:
    """Validate and load a WAV file as a 1-D float32 array in [-1, 1]."""
    validate_wav(path)
    audio, _ = sf.read(str(path), dtype="float32")
    return audio
