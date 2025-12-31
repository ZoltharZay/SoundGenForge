from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .audiobox_aesthetics import AudioboxAestheticsScorer, AestheticsScore
from .flamingo3 import AudioFlamingo3Scorer, FlamingoScore

@dataclass(frozen=True)
class EvalResult:
    audiobox: Optional[AestheticsScore] = None
    flamingo: Optional[FlamingoScore] = None

class Evaluator:
    def __init__(self, use_audiobox: bool, use_flamingo3: bool) -> None:
        self._audiobox = AudioboxAestheticsScorer() if use_audiobox else None
        self._flamingo = AudioFlamingo3Scorer() if use_flamingo3 else None
        # State to track the attempt index (0-4)
        self._attempt_counter = 0

    def evaluate(self, audio_path: Path, gen_prompt: str) -> EvalResult:
        ab = self._audiobox.score_file(audio_path) if self._audiobox else None
        fl = self._flamingo.score_file(audio_path, gen_prompt) if self._flamingo else None
        return EvalResult(audiobox=ab, flamingo=fl)

    def passes(
        self,
        res: EvalResult,
        pq_thresh: float = 6.0,
        ce_thresh: float = 6.0,
        flamingo_thresh: float = 6.0
    ) -> bool:
        ok = True
        if res.audiobox is not None:
            ok = ok and (res.audiobox.pq >= pq_thresh) and (res.audiobox.ce >= ce_thresh)
        if res.flamingo is not None:
            ok = ok and (res.flamingo.score >= flamingo_thresh)
        return ok

    def feedback_sentence(self, res: EvalResult) -> str:
        # Variations for Quality (PQ < 6.0)
        pq_variations = [
            "Reduce noise and reverb; make the transient cleaner and louder.",
            "Minimize background hiss and sharpen the audio",
            "Focus on a drier recording with a punchier, cleaner transient.",
            "Ensure the audio is clearer; boost the impact and cut the reverb.",
            "Aim for a cleaner signal-to-noise ratio and a much sharper start."
        ]

        # Variations for Style (Default)
        style_variations = [
            "Make the sound clearer, shorter, and more game-like.",
            "Aim for a punchier, more stylized effect suitable for a game.",
            "Shorten the duration and emphasize the core gameplay sound.",
            "Focus on making the sound more responsive and distinct for gaming.",
            "Ensure the audio is concise and has a clear, game-oriented texture."
        ]

        # 1. Prefer Flamingo feedback if present
        if res.flamingo is not None and res.flamingo.feedback:
            return res.flamingo.feedback.strip()

        # 2. Check Audiobox PQ Score
        if res.audiobox is not None and res.audiobox.pq < 6.0:
            feedback = pq_variations[self._attempt_counter]
            self._attempt_counter = (self._attempt_counter + 1) % 5
            return feedback

        # 3. Default "Game-like" advice
        feedback = style_variations[self._attempt_counter]
        self._attempt_counter = (self._attempt_counter + 1) % 5
        return feedback
