from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass(frozen=True)
class GenerationResult:
    audio_path: Path
    format: str  # e.g., wav, mp3
    sample_rate: Optional[int] = None

class BaseGenerator:
    name: str

    def generate(self, *, prompt: str, duration_s: float, out_path: Path, loop: bool = False, seed: Optional[int] = None) -> GenerationResult:
        raise NotImplementedError
