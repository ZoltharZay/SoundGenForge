from __future__ import annotations
from pathlib import Path
import subprocess
import numpy as np

def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)

def ffmpeg_to_wav(src: Path, dst: Path, sample_rate: int = 44100) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-i", str(src),
        "-ar", str(sample_rate),
        "-ac", "2",
        str(dst)
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def trim_or_pad_int16(audio_int16: np.ndarray, target_samples: int) -> np.ndarray:
    # audio_int16: shape (channels, samples) or (samples,)
    if audio_int16.ndim == 1:
        audio_int16 = audio_int16[None, :]
    ch, n = audio_int16.shape
    if n == target_samples:
        return audio_int16
    if n > target_samples:
        return audio_int16[:, :target_samples]
    pad = np.zeros((ch, target_samples - n), dtype=np.int16)
    return np.concatenate([audio_int16, pad], axis=1)
