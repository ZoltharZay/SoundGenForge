from __future__ import annotations
from pathlib import Path
from typing import Optional
from ..utils.text import shorten_for_elevenlabs

from .base import BaseGenerator, GenerationResult
from ..utils.audio_io import write_bytes

class ElevenLabsGenerator(BaseGenerator):
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.elevenlabs.io",
        output_format: str = "mp3_44100_128",
        model_id: str = "eleven_text_to_sound_v2",
        prompt_influence: float = 0.3,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.output_format = output_format
        self.model_id = model_id
        self.prompt_influence = prompt_influence

        try:
            from elevenlabs import ElevenLabs  # type: ignore
        except Exception as e:
            raise RuntimeError("Install ElevenLabs SDK: pip install elevenlabs") from e

        # The SDK handles auth; base_url is optional but user requested it.
        self._client = ElevenLabs(api_key=api_key, base_url=base_url)

    def generate(self, *, prompt: str, duration_s: float, out_path: Path, loop: bool = False, seed: Optional[int] = None) -> GenerationResult:
        # ElevenLabs sound effects API streams MP3 bytes.
        short_prompt = shorten_for_elevenlabs(prompt)
        audio_iter = self._client.text_to_sound_effects.convert(
            text=short_prompt,
            output_format=self.output_format,
            loop=bool(loop),
            duration_seconds=float(duration_s),
            prompt_influence=float(self.prompt_influence),
            model_id=self.model_id,
        )
        data = b"".join(audio_iter)
        write_bytes(out_path, data)
        return GenerationResult(audio_path=out_path, format="mp3", sample_rate=None)
