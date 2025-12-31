from __future__ import annotations

from pathlib import Path
from typing import Optional
import torch

from .prompted_judge_base import BasePromptedAudioJudge, PromptStyle


class Qwen3OmniJudge(BasePromptedAudioJudge):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-Omni",
        device: Optional[str] = None,
        torch_dtype: Optional[str] = None,  # "bfloat16" | "float16" | "float32"
        max_new_tokens: int = 256,
        prompt_style: PromptStyle = "guided",
        trust_remote_code: bool = True,
        attn_implementation: str = "flash_attention_2",
        use_fast_processor: bool = False,
    ) -> None:
        super().__init__(prompt_style=prompt_style)

        from transformers import (  # type: ignore
            Qwen3OmniMoeForConditionalGeneration,
            Qwen3OmniMoeProcessor,
            GenerationConfig,
        )
        from qwen_omni_utils import process_mm_info  # type: ignore

        self._process_mm_info = process_mm_info
        self.GenerationConfig = GenerationConfig

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device  # keep as string

        if torch_dtype is None:
            dtype = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        else:
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
        self.torch_dtype = dtype

        self.model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map="auto" if str(device).startswith("cuda") else None,
            attn_implementation=attn_implementation if str(device).startswith("cuda") else None,
            trust_remote_code=trust_remote_code,
        ).eval()

        self.processor = Qwen3OmniMoeProcessor.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code,
            use_fast=use_fast_processor,
        )

        self.max_new_tokens = int(max_new_tokens)

        # --- Critical workaround for dtype mismatch inside audio tower ---
        # Some internal audio feature steps produce float32 tensors even if inputs are bf16,
        # then conv2d1 has bf16 bias -> RuntimeError. We cast conv2d1 input to conv dtype.
        self._install_audio_cast_hook()

    def _install_audio_cast_hook(self) -> None:
        try:
            audio_tower = getattr(self.model, "audio_tower", None)
            if audio_tower is None:
                return

            conv = getattr(audio_tower, "conv2d1", None)
            if conv is None:
                return

            # Get conv parameter dtype (bf16/fp16/fp32)
            conv_dtype = next(conv.parameters()).dtype

            def _pre_hook(module, inputs):
                if not inputs:
                    return inputs
                x = inputs[0]
                if isinstance(x, torch.Tensor) and torch.is_floating_point(x) and x.dtype != conv_dtype:
                    x = x.to(dtype=conv_dtype)
                    return (x,) + tuple(inputs[1:])
                return inputs

            # register_forward_pre_hook signature differs slightly across torch versions
            try:
                conv.register_forward_pre_hook(_pre_hook, with_kwargs=False)
            except TypeError:
                conv.register_forward_pre_hook(_pre_hook)

        except Exception:
            # If anything about hooks fails, we just do nothing (won't hide the error, but avoids crashing init)
            return

    def _pick_input_device(self) -> torch.device:
        """
        For device_map='auto' models, next(self.model.parameters()).device can be CPU/meta.
        We pick a sane CUDA device for input tensors.
        """
        # If user asked for CPU, honor it
        if not str(self.device).startswith("cuda"):
            return torch.device("cpu")

        # Prefer thinker device if available
        thinker = getattr(self.model, "thinker", None)
        if thinker is not None:
            try:
                d = next(thinker.parameters()).device
                if d.type == "cuda":
                    return d
            except Exception:
                pass

        # Fall back to hf_device_map if present
        dev_map = getattr(self.model, "hf_device_map", None)
        if isinstance(dev_map, dict):
            for _, d in dev_map.items():
                try:
                    td = torch.device(str(d))
                    if td.type == "cuda":
                        return td
                except Exception:
                    continue

        # Final fallback
        return torch.device("cuda")

    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        mnt = int(max_new_tokens) if max_new_tokens else self.max_new_tokens

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": str(audio_path)},
                    {"type": "text", "text": text},
                ],
            }
        ]

        templated = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = self._process_mm_info(conversation, use_audio_in_video=False)

        batch = self.processor(
            text=templated,
            audio=audios,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            use_audio_in_video=False,
        )

        # Convert to a plain dict to avoid any BatchFeature quirks
        inputs = {k: v for k, v in batch.items()}

        target_device = self._pick_input_device()
        target_dtype = self.torch_dtype

        # Move & cast:
        # - integer tensors (input_ids/attention_mask) -> device (no dtype change)
        # - float tensors -> device + cast to target_dtype
        def _move_cast(obj):
            if isinstance(obj, torch.Tensor):
                obj = obj.to(target_device)
                if torch.is_floating_point(obj):
                    obj = obj.to(dtype=target_dtype)
                return obj
            if isinstance(obj, dict):
                return {k: _move_cast(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                out = [_move_cast(x) for x in obj]
                return type(obj)(out)
            return obj

        inputs = _move_cast(inputs)

        pad_token_id = self.processor.tokenizer.eos_token_id
        gen_cfg = self.GenerationConfig(
            max_new_tokens=mnt,
            do_sample=False,
            pad_token_id=pad_token_id,
            eos_token_id=pad_token_id,
        )

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, generation_config=gen_cfg)

        prompt_len = inputs["input_ids"].size(1)
        response_ids = generated_ids[:, prompt_len:]
        out = self.processor.batch_decode(
            response_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return out.strip()

