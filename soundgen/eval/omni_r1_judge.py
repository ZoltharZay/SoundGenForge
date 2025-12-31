from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch

from .prompted_judge_base import BasePromptedAudioJudge, PromptStyle


QWEN_SYS_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


class OmniR1Judge(BasePromptedAudioJudge):
    def __init__(
        self,
        model_id: str = "Haoz0206/Omni-R1",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,  # "bfloat16" | "float16" | "float32"
        max_new_tokens: int = 256,
        prompt_style: PromptStyle = "guided",
        trust_remote_code: bool = True,
        attn_implementation: str = "flash_attention_2",
    ) -> None:
        super().__init__(prompt_style=prompt_style)

        from transformers import (  # type: ignore
            Qwen2_5OmniThinkerForConditionalGeneration,
            Qwen2_5OmniProcessor,
        )
        from qwen_omni_utils import process_mm_info  # type: ignore

        self._process_mm_info = process_mm_info

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            dtype = torch.bfloat16 if device == "cuda" else torch.float32
        else:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
        self.torch_dtype = dtype

        self.model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            attn_implementation=attn_implementation if device == "cuda" else None,
            trust_remote_code=trust_remote_code,
        ).eval()

        self.processor = Qwen2_5OmniProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
        )

        self.generation_config = self.model.generation_config
        self.generation_config.max_new_tokens = int(max_new_tokens)
        self.generation_config.do_sample = False

    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        # Keep per-call override without mutating shared config too aggressively
        gen_cfg = self.generation_config
        gen_cfg.max_new_tokens = int(max_new_tokens)

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": QWEN_SYS_PROMPT}]},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": text},
                ],
            },
        ]

        templated_text = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )

        audio_input, image_input, video_input = self._process_mm_info(conversation, use_audio_in_video=False)

        inputs = self.processor(
            text=templated_text,
            images=image_input,
            audio=audio_input,   # IMPORTANT: 'audio=' (not 'audios='), per your snippet
            videos=video_input,
            return_tensors="pt",
            do_resize=True,
        )

        # Move tensors
        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
                if torch.is_floating_point(inputs[k]):
                    inputs[k] = inputs[k].to(self.torch_dtype)

        with torch.inference_mode():
            generated_ids = self.model.generate(**inputs, generation_config=gen_cfg)

        prompt_len = inputs["input_ids"].size(1)
        completion_ids = generated_ids[:, prompt_len:]
        out = self.processor.batch_decode(completion_ids, skip_special_tokens=True)[0]
        return out.strip()
