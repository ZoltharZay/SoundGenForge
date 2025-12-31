from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torchaudio

@dataclass(frozen=True)
class AestheticsScore:
    ce: float
    pq: float

class AudioboxAestheticsScorer:
    """Thin wrapper around Meta's audiobox_aesthetics predictor.

    Requires:
      pip install audiobox_aesthetics
    """
    def __init__(self) -> None:
        try:
            from audiobox_aesthetics.infer import initialize_predictor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "audiobox_aesthetics is not installed. Install with: pip install audiobox_aesthetics"
            ) from e
        self._predictor = initialize_predictor()

    def score_file(self, audio_path: Path) -> AestheticsScore:
        # Predictor supports file path dicts directly.
        out = self._predictor.forward([{"path": str(audio_path)}])
        # out is typically a list-like of dicts, or a dict; normalize:
        if isinstance(out, list) and len(out) > 0 and isinstance(out[0], dict):
            d = out[0]
        elif isinstance(out, dict):
            d = out
        else:
            # best-effort: try treating it as list of JSON strings
            raise RuntimeError(f"Unexpected predictor output type: {type(out)} / {out}")
        ce = float(d.get("CE"))
        pq = float(d.get("PQ"))
        return AestheticsScore(ce=ce, pq=pq)
