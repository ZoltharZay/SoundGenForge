from __future__ import annotations

from pathlib import Path
from typing import Optional

import random
import numpy as np
import torch

from ..utils.onset import crop_sfx_1s
from ..utils.text import shorten_for_elevenlabs
from ..utils.audio_io import ffmpeg_to_wav  # <-- reuse your known-good normalization
from .base import BaseGenerator, GenerationResult


def save_wav_compat(path: Path, audio: torch.Tensor, sr: int) -> None:
    """
    Save WAV without torchaudio/torchcodec. Uses soundfile (libsndfile).
    audio: (C, N) float tensor in [-1, 1]
    """
    audio = audio.detach().cpu().to(torch.float32)
    audio = torch.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1, 1)

    # (C, N) -> (N, C)
    data = audio.transpose(0, 1).numpy()

    import soundfile as sf  # pip install soundfile
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype="PCM_16")


def seed_everything(seed: int) -> None:
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TangoFluxGenerator(BaseGenerator):
    name = "tangoflux"

    def __init__(self, model_name: str = "declare-lab/TangoFlux", steps: int = 50) -> None:
        try:
            from tangoflux import TangoFluxInference  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TangoFlux is not installed. Install with: pip install git+https://github.com/declare-lab/TangoFlux"
            ) from e

        self._TangoFluxInference = TangoFluxInference
        self.model_name = model_name
        self.steps = int(steps)
        self._model = None

    def _lazy_model(self):
        if self._model is None:
            self._model = self._TangoFluxInference(name=self.model_name)

            # Best-effort CUDA move (kept minimal; TangoFluxInference may manage devices internally)
            if torch.cuda.is_available():
                dev = torch.device("cuda")
                if hasattr(self._model, "to"):
                    try:
                        self._model = self._model.to(dev)
                    except Exception:
                        pass
                for attr in ("pipe", "pipeline", "model"):
                    if hasattr(self._model, attr):
                        sub = getattr(self._model, attr)
                        if hasattr(sub, "to"):
                            try:
                                sub.to(dev)
                            except Exception:
                                pass

        return self._model

    @staticmethod
    def _ensure_tensor_audio(audio) -> torch.Tensor:
        """
        Normalize TangoFlux output to torch.Tensor (C, N) on CPU float32.
        """
        if isinstance(audio, torch.Tensor):
            a = audio
        else:
            a = torch.tensor(audio)

        if a.ndim == 1:
            a = a.unsqueeze(0)  # (1, N)

        a = a.detach().cpu().to(torch.float32)
        a = torch.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

        # If empty or extremely short, pad to at least 1 sample to avoid decoder edge cases
        if a.shape[1] < 1:
            a = torch.zeros((a.shape[0], 1), dtype=torch.float32)

        # Clamp to a safe range
        a = a.clamp(-1, 1)
        return a

    def _save_ffmpeg_normalized_wav(self, out_path: Path, audio: torch.Tensor, sr: int) -> None:
        """
        TangoFlux-only fix: always normalize final WAV through ffmpeg into PCM WAV.
        This avoids weird container/codec issues that can make Audiobox scoring crash.
        """
        tmp = out_path.with_suffix(".tmp.wav")

        # 1) write via soundfile (no torchaudio/torchcodec)
        save_wav_compat(tmp, audio, sr)

        # 2) normalize via ffmpeg into the *real* output file (PCM, correct SR)
        #    Uses your project's ffmpeg_to_wav wrapper.
        try:
            ffmpeg_to_wav(tmp, out_path, sample_rate=sr)
        finally:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:
                pass

    def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        out_path: Path,
        loop: bool = False,
        seed: Optional[int] = None
    ) -> GenerationResult:
        model = self._lazy_model()

        current_seed = seed if seed is not None else random.randint(0, 1_000_000)
        seed_everything(current_seed)

        # Keep your existing shared prompt shortening behavior
        prompt = shorten_for_elevenlabs(prompt)

        sr = 44100
        duration_s = float(duration_s)

        if duration_s <= 1.05:
            internal_duration = 5.0
            raw = model.generate(prompt, steps=int(self.steps), duration=internal_duration)
            audio = self._ensure_tensor_audio(raw)

            # Crop onset to 1.0s (your original behavior)
            audio = crop_sfx_1s(audio, sr, target_s=1.0)
        else:
            raw = model.generate(prompt, steps=int(self.steps), duration=duration_s)
            audio = self._ensure_tensor_audio(raw)

        out_path.parent.mkdir(parents=True, exist_ok=True)

        # -----------------------------
        # KEY FIX: normalize TangoFlux wav through ffmpeg
        # -----------------------------
        self._save_ffmpeg_normalized_wav(out_path, audio, sr)

        return GenerationResult(audio_path=out_path, format="wav", sample_rate=sr)

