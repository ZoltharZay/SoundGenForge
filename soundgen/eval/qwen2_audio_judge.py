from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch

from .prompted_judge_base import BasePromptedAudioJudge, PromptStyle


class Qwen2AudioJudge(BasePromptedAudioJudge):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-Audio",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,  # "float16" | "bfloat16" | "float32"
        max_new_tokens: int = 256,
        prompt_style: PromptStyle = "guided",
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__(prompt_style=prompt_style)

        from transformers import (  # type: ignore
            Qwen2AudioForConditionalGeneration,
            AutoProcessor,
            GenerationConfig,
        )
        from transformers.pipelines.audio_utils import ffmpeg_read  # type: ignore

        self.ffmpeg_read = ffmpeg_read
        self.GenerationConfig = GenerationConfig

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        if torch_dtype is None:
            dtype = torch.float16 if device == "cuda" else torch.float32
        else:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
        self.torch_dtype = dtype

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_id,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        ).eval()

        self.max_new_tokens = int(max_new_tokens)

    def _read_bytes(self, p: Path) -> bytes:
        return p.read_bytes()

    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        mnt = int(max_new_tokens) if max_new_tokens else self.max_new_tokens

        # Conversation template with audio placeholder (model expects audio slots in template)
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": text}]},
        ]
        templated_text = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True, tokenize=False
        )

        # Decode audio bytes via ffmpeg_read (as in your example)
        raw = self._read_bytes(audio_path)
        input_audio = self.ffmpeg_read(raw, sampling_rate=self.processor.feature_extractor.sampling_rate)

        inputs = self.processor(
            text=templated_text,
            audio=input_audio,
            sampling_rate=self.processor.feature_extractor.sampling_rate,
            return_tensors="pt",
            padding=True,
        )

        for k, v in list(inputs.items()):
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(self.model.device)
                if torch.is_floating_point(inputs[k]):
                    inputs[k] = inputs[k].to(self.torch_dtype)

        pad_token_id = getattr(getattr(self.processor, "tokenizer", self.processor), "pad_token_id", None)
        eos_token_id = getattr(getattr(self.processor, "tokenizer", self.processor), "eos_token_id", None)
        gen_cfg = self.GenerationConfig(
            max_new_tokens=mnt,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=gen_cfg)

        prompt_len = inputs["input_ids"].size(1)
        response_ids = generated_ids[:, prompt_len:]
        generated_text = self.processor.batch_decode(response_ids, skip_special_tokens=True)[0]

        # Light cleanup consistent with your example approach
        parts = generated_text.split("assistant\n")
        out = parts[-1].strip() if len(parts) > 1 else generated_text.strip()
        return out
