from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class OnsetConfig:
    # Envelope + threshold
    smooth_ms: float = 12.0           # moving average window for envelope smoothing
    noise_ms: float = 250.0           # initial window used to estimate noise floor
    thresh_mult: float = 8.0          # threshold = max(abs_thresh, noise_floor * mult)
    abs_thresh: float = 0.01          # absolute fallback threshold (audio assumed in [-1,1])

    # Onset validation
    min_hold_ms: float = 25.0         # must stay above threshold this long to count as onset
    preroll_ms: float = 10.0          # start crop slightly before onset


def _mono(x: torch.Tensor) -> torch.Tensor:
    # x: (C, N) or (N,)
    if x.ndim == 1:
        return x
    return x.mean(dim=0)


def _moving_average(x: torch.Tensor, win: int) -> torch.Tensor:
    if win <= 1:
        return x
    # simple 1D conv with ones
    w = torch.ones(win, device=x.device, dtype=x.dtype) / float(win)
    x2 = x.unsqueeze(0).unsqueeze(0)  # (1,1,N)
    y = torch.nn.functional.conv1d(x2, w.view(1, 1, -1), padding=win // 2)
    return y.squeeze(0).squeeze(0)


def find_onset_sample(audio: torch.Tensor, sr: int, cfg: OnsetConfig) -> Optional[int]:
    """
    Returns onset sample index (int) or None if not found.
    audio: (C,N) torch tensor on CPU
    """
    x = _mono(audio).to(torch.float32)
    x = x.clamp(-1, 1)

    env = x.abs()
    smooth = max(1, int(round(cfg.smooth_ms * sr / 1000.0)))
    env_s = _moving_average(env, smooth)

    noise_n = max(1, int(round(cfg.noise_ms * sr / 1000.0)))
    noise_floor = float(env_s[: min(noise_n, env_s.numel())].median())
    thr = max(cfg.abs_thresh, noise_floor * cfg.thresh_mult)

    hold_n = max(1, int(round(cfg.min_hold_ms * sr / 1000.0)))
    pre_n = max(0, int(round(cfg.preroll_ms * sr / 1000.0)))

    # Find first index where env stays above threshold for hold_n samples
    above = env_s > thr
    if above.sum().item() == 0:
        return None

    # run-length check
    run = 0
    for i in range(above.numel()):
        run = run + 1 if bool(above[i]) else 0
        if run >= hold_n:
            onset = i - hold_n + 1
            return max(0, onset - pre_n)

    return None


def crop_sfx_1s(
    audio: torch.Tensor,
    sr: int,
    target_s: float = 1.0,
    onset_cfg: Optional[OnsetConfig] = None,
) -> torch.Tensor:
    """
    Crops a deterministic 1s segment starting at detected onset.
    Fallback: max-energy 1s window if onset not found.
    audio: (C,N)
    """
    if onset_cfg is None:
        onset_cfg = OnsetConfig()

    target_samples = int(round(target_s * sr))
    if audio.shape[1] <= target_samples:
        # pad if short
        if audio.shape[1] < target_samples:
            pad = torch.zeros((audio.shape[0], target_samples - audio.shape[1]), dtype=audio.dtype)
            return torch.cat([audio, pad], dim=1)
        return audio

    onset = find_onset_sample(audio, sr, onset_cfg)

    if onset is not None:
        end = min(audio.shape[1], onset + target_samples)
        seg = audio[:, onset:end]
        if seg.shape[1] < target_samples:
            pad = torch.zeros((seg.shape[0], target_samples - seg.shape[1]), dtype=seg.dtype)
            seg = torch.cat([seg, pad], dim=1)
        return seg

    # Fallback: max-energy window (deterministic)
    hop = max(1, int(0.02 * sr))  # 20ms
    best_i = 0
    best_e = -1.0
    i = 0
    while i + target_samples <= audio.shape[1]:
        seg = audio[:, i:i + target_samples]
        e = float(torch.mean(seg.to(torch.float32) ** 2))
        if e > best_e:
            best_e = e
            best_i = i
        i += hop
    return audio[:, best_i:best_i + target_samples]

