from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import torch

@dataclass(frozen=True)
class Flamingo3Score:
    score_0_10: float
    feedback: str
    raw: str


class AudioFlamingo3Scorer:
    def __init__(self, model_id: str = "nvidia/audio-flamingo-3-hf") -> None:
        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "AudioFlamingo3ForConditionalGeneration is unavailable. "
                "Install/upgrade Transformers (>=4.46.0 recommended for AF3)."
            ) from e

        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(model_id, device_map="auto")
        self.model.eval()

    @staticmethod
    def _parse_score(text: str) -> float:
        # Try SCORE: x first
        m = re.search(r"(?i)\bscore\b\s*[:=]\s*(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)", text)
        if not m:
            # fallback: first number in range
            m = re.search(r"\b(10(?:\.0+)?|[0-9](?:\.[0-9]+)?)\b", text)
        if not m:
            return float("nan")
        v = float(m.group(1))
        return max(0.0, min(10.0, v))

    @staticmethod
    def _parse_feedback(text: str) -> str:
        m = re.search(r"(?i)\bfeedback\b\s*[:=]\s*(.+)", text)
        if m:
            return m.group(1).strip().splitlines()[0][:240]
        # fallback: first non-empty line
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:240]
        return ""

    def score_file(self, audio_path: Path, generation_prompt: str) -> Flamingo3Score:
        instruction = (
            "You are rating a generated game sound effect against its generation prompt.\n"
            f"PROMPT: {generation_prompt}\n"
            "Output:\n"
            "SCORE: <0-10>\n"
            "FEEDBACK: <one short sentence to improve the next attempt>\n"
        )

        conversation = [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "path": str(audio_path)},
            ],
        }]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=96)

        decoded = self.processor.batch_decode(
            outputs[:, inputs.input_ids.shape[1]:],
            skip_special_tokens=True
        )[0].strip()

        score = self._parse_score(decoded)
        feedback = self._parse_feedback(decoded)

        if not feedback:
            feedback = "Make the sound shorter, clearer, and more impact-focused."

        return Flamingo3Score(score_0_10=score, feedback=feedback, raw=decoded)

