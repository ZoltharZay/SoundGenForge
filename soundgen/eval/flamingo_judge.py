from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch

from .prompted_judge_base import BasePromptedAudioJudge, PromptStyle


class AudioFlamingo3Judge(BasePromptedAudioJudge):
    def __init__(
        self,
        model_id: str = "nvidia/audio-flamingo-3-hf",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,  # "float16" | "bfloat16" | "float32"
        prompt_style: PromptStyle = "guided",
    ) -> None:
        super().__init__(prompt_style=prompt_style)

        try:
            from transformers import AudioFlamingo3ForConditionalGeneration, AutoProcessor  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "AudioFlamingo3 classes are not available in your installed transformers. "
                "Upgrade transformers to a recent build that includes AudioFlamingo3."
            ) from e

        self.processor = AutoProcessor.from_pretrained(model_id)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        dtype = None
        if torch_dtype:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]

        if self.device.type == "cuda":
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id,
                device_map="auto",
                torch_dtype=dtype,
            )
        else:
            self.model = AudioFlamingo3ForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=dtype,
            ).to(self.device)

        self.model.eval()

    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": text},
                    {"type": "audio", "path": str(audio_path)},
                ],
            }
        ]

        inputs = self.processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        model_device = self.model.device
        model_dtype = next(self.model.parameters()).dtype

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                v = v.to(model_device)
                if torch.is_floating_point(v):
                    v = v.to(model_dtype)
                inputs[k] = v

        with torch.no_grad():
            out_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        decoded = self.processor.batch_decode(
            out_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0]
        return decoded.strip()

