from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
from einops import rearrange

from ..utils.onset import crop_sfx_1s
from ..utils.text import shorten_for_elevenlabs
from .base import BaseGenerator, GenerationResult


def seed_everything(seed: int) -> None:
    seed = int(seed) & 0xFFFFFFFF
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _try_import_audiox():
    """
    Import stable_audio_tools lazily so other generators can run without AudioX deps.
    """
    from stable_audio_tools import get_pretrained_model  # type: ignore
    from stable_audio_tools.inference.generation import generate_diffusion_cond  # type: ignore
    return get_pretrained_model, generate_diffusion_cond


def _save_wav(path: Path, audio: torch.Tensor, sr: int) -> None:
    """
    Save WAV robustly on aarch64 setups where torchaudio may require torchcodec.
    audio: (C, N) float tensor in [-1, 1]
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = audio.detach().cpu().to(torch.float32).clamp(-1, 1)

    try:
        torchaudio.save(str(path), audio, sr)
        return
    except Exception:
        # Fallback: soundfile (pip install soundfile)
        import soundfile as sf

        data = audio.transpose(0, 1).numpy()  # (N, C)
        sf.write(str(path), data, sr, subtype="PCM_16")


class AudioXGenerator(BaseGenerator):
    """
    AudioX generator for environments where the checkpoint's conditioner still expects
    video + audio keys. We provide:
      - dummy black video: 224x224, T=50 frames (matches your working configuration)
      - silent audio prompt shaped as (B, C, N)
      - seconds_total fixed to 10.0 for conditioning stability

    SFX behavior:
      - If duration_s <= 1.05: always generate one full 10s chunk, then onset-crop 1s.

    BGM/longer:
      - stitch chunks as needed, then trim/pad to target duration.
    """

    name = "audiox"

    def __init__(
        self,
        hf_model_id: str = "HKUSTAudio/AudioX",
        steps: int = 50,
        cfg_scale: float = 7.0,
        sampler_type: str = "dpmpp-3m-sde",
        sigma_min: float = 0.3,
        sigma_max: float = 500.0,
        device: Optional[str] = None,
        conditioning_seconds: float = 10.0,  # keep conditioner shapes stable
        dummy_video_frames: int = 50,        # IMPORTANT
        dummy_video_size: int = 224,         # IMPORTANT
    ) -> None:
        try:
            self._get_pretrained_model, self._generate_diffusion_cond = _try_import_audiox()
        except Exception as e:
            raise RuntimeError(
                "AudioX/stable_audio_tools is not available in this environment. "
                "Install AudioX deps in this env (stable_audio_tools, transformers, etc.)."
            ) from e

        self.hf_model_id = hf_model_id
        self.steps = int(steps)
        self.cfg_scale = float(cfg_scale)
        self.sampler_type = str(sampler_type)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.device_override = device

        self.conditioning_seconds = float(conditioning_seconds)
        self.dummy_video_frames = int(dummy_video_frames)
        self.dummy_video_size = int(dummy_video_size)

        self._model = None
        self._model_config = None

        # cache dummy video prompt on the active device
        self._cached_video_prompt = None  # type: Optional[list]
        self._cached_video_device = None  # type: Optional[torch.device]

    def _device(self) -> torch.device:
        if self.device_override:
            return torch.device(self.device_override)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _lazy_model(self) -> Tuple[torch.nn.Module, dict]:
        if self._model is None or self._model_config is None:
            model, cfg = self._get_pretrained_model(self.hf_model_id)
            self._model = model
            self._model_config = cfg
        return self._model, self._model_config

    def _get_cached_dummy_video_prompt(self, device: torch.device) -> list:
        """
        Format expected by stable_audio_tools:
            "video_prompt": [video_tensor.unsqueeze(0)]
        where video_tensor is (T, C, H, W) and .unsqueeze(0) makes (1, T, C, H, W)
        """
        if self._cached_video_prompt is not None and self._cached_video_device == device:
            return self._cached_video_prompt

        T = self.dummy_video_frames
        S = self.dummy_video_size

        video_tensor = torch.zeros((T, 3, S, S), dtype=torch.float32, device=device)
        video_prompt = [video_tensor.unsqueeze(0)]

        self._cached_video_prompt = video_prompt
        self._cached_video_device = device
        return video_prompt

    def _gen_one_chunk(self, prompt: str, seed: Optional[int]) -> torch.Tensor:
        """
        Generate one chunk (conditioning_seconds, typically 10s).
        Returns (C, N) on CPU.
        """
        if seed is not None:
            seed_everything(seed)

        model, cfg = self._lazy_model()
        device = self._device()
        device_str = "cuda" if device.type == "cuda" else device.type

        model = model.to(device)
        model.eval()

        sample_rate = int(cfg["sample_rate"])
        sample_size = int(cfg["sample_size"])

        seconds_total = float(self.conditioning_seconds)
        seconds_start = 0.0

        # silent audio prompt must be (B, C, N)
        n = int(round(seconds_total * sample_rate))
        audio_prompt = torch.zeros((1, 1, n), dtype=torch.float32, device=device)

        conditioning = [{
            "video_prompt": self._get_cached_dummy_video_prompt(device),
            "text_prompt": prompt,
            "audio_prompt": audio_prompt,
            "seconds_start": seconds_start,
            "seconds_total": seconds_total,
        }]

        with torch.inference_mode():
            out = self._generate_diffusion_cond(
                model,
                steps=self.steps,
                cfg_scale=self.cfg_scale,
                conditioning=conditioning,
                sample_size=sample_size,
                sigma_min=self.sigma_min,
                sigma_max=self.sigma_max,
                sampler_type=self.sampler_type,
                device=device_str,  # <-- key for CUDA execution
                seed=-1 if seed is None else int(seed),
            )

        out = rearrange(out, "b d n -> d (b n)").to(torch.float32)

        # normalize safely
        out = out / torch.max(torch.abs(out)).clamp(min=1e-6)
        out = out.clamp(-1, 1)
        return out.detach().cpu()

    def generate(
        self,
        *,
        prompt: str,
        duration_s: float,
        out_path: Path,
        loop: bool = False,
        seed: Optional[int] = None
    ) -> GenerationResult:
        _, cfg = self._lazy_model()
        sample_rate = int(cfg["sample_rate"])

        prompt = shorten_for_elevenlabs(prompt)
        duration_s = float(duration_s)
        chunk_s = float(self.conditioning_seconds)

        # --- SFX path: always generate full chunk (10s), then onset-crop 1s ---
        if duration_s <= 1.05:
            audio_long = self._gen_one_chunk(prompt, seed=seed)  # (C, ~10s)
            audio = crop_sfx_1s(audio_long, sample_rate, target_s=1.0)

        # --- Non-SFX path: stitch then trim/pad from start ---
        else:
            if duration_s > chunk_s + 1e-6:
                n_chunks = int(math.ceil(duration_s / chunk_s))
                chunks = []
                for i in range(n_chunks):
                    s = None if seed is None else (seed + i)
                    chunks.append(self._gen_one_chunk(prompt, seed=s))
                audio = torch.cat(chunks, dim=1)
            else:
                audio = self._gen_one_chunk(prompt, seed=seed)

            target_samples = int(round(duration_s * sample_rate))
            audio = audio[:, :target_samples]
            if audio.shape[1] < target_samples:
                pad = torch.zeros((audio.shape[0], target_samples - audio.shape[1]), dtype=audio.dtype)
                audio = torch.cat([audio, pad], dim=1)

        _save_wav(out_path, audio, sample_rate)
        return GenerationResult(audio_path=out_path, format="wav", sample_rate=sample_rate)

