from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch

from .prompted_judge_base import BasePromptedAudioJudge, PromptStyle


class KimiAudioJudge(BasePromptedAudioJudge):
    def __init__(
        self,
        model_path: str,
        prompt_style: PromptStyle = "guided",
    ) -> None:
        super().__init__(prompt_style=prompt_style)

        try:
            from kimia_infer.api.kimia import KimiAudio  # type: ignore
        except ImportError as e:
            raise ImportError(
                "KimiAudio package not found. Install via:\n"
                "pip install git+https://github.com/MoonshotAI/Kimi-Audio.git"
            ) from e

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = KimiAudio(
            model_path=model_path,
            load_detokenizer=True,
        )

        # Sampling params (from your example defaults)
        self.sampling_params = {
            "audio_temperature": 0.8,
            "audio_top_k": 10,
            "text_temperature": 0.0,
            "text_top_k": 5,
            "audio_repetition_penalty": 1.0,
            "audio_repetition_window_size": 64,
            "text_repetition_penalty": 1.0,
            "text_repetition_window_size": 16,
        }

    def _call_kimi(self, messages: list) -> str:
        try:
            _, text = self.model.generate(
                messages,
                **self.sampling_params,
                output_type="text",
            )
            return text
        except Exception as e:
            return f"Error during Kimi model generation: {e}"

    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        # Kimi API in your example doesn't use max_new_tokens directly.
        messages = [
            {"role": "user", "message_type": "text", "content": text},
            {"role": "user", "message_type": "audio", "content": str(audio_path)},
        ]
        out = self._call_kimi(messages)
        return out.strip()
