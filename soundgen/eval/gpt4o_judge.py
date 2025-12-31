from __future__ import annotations

from pathlib import Path
from typing import Optional
import base64
import subprocess
import tempfile

from .prompted_judge_base import BasePromptedAudioJudge, PromptStyle


class GPT4oJudge(BasePromptedAudioJudge):
    """
    OpenAI audio judge for models like: "gpt-audio-mini"

    - Converts any input audio -> WAV PCM16 mono 16kHz
    - Sends as input_audio (format="wav")
    - Retries once if model responds as if audio is missing / claims it cannot listen
    - If still refusal-like, returns deterministic fallback "Score: 0"
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: Optional[str],
        model_name: str,  # e.g. "gpt-audio-mini"
        prompt_style: PromptStyle = "guided",
        max_tokens: int = 256,
        sample_rate_hz: int = 16000,
        channels: int = 1,
    ) -> None:
        super().__init__(prompt_style=prompt_style)

        from openai import OpenAI  # type: ignore

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model_name = model_name
        self.max_tokens = int(max_tokens)
        self.sample_rate_hz = int(sample_rate_hz)
        self.channels = int(channels)

    def _convert_to_wav_pcm16(self, src: Path) -> Path:
        tmp_dir = Path(tempfile.mkdtemp(prefix="gpt_audio_judge_"))
        out_wav = tmp_dir / "audio.wav"

        # -af "apad=whole_len=32000" adds silence to the end to reach 2 seconds (at 16kHz)
        # 16000 samples/sec * 2 seconds = 32000 samples
        min_samples = self.sample_rate_hz * 2 

        cmd = [
            "ffmpeg",
            "-y",
            "-vn",
            "-i", str(src),
            "-ar", str(self.sample_rate_hz),
            "-ac", str(self.channels),
            "-c:a", "pcm_s16le",
            "-af", f"apad=whole_len={min_samples}", # Ensure minimum duration
            "-f", "wav",
            "-bitexact", # Removes non-standard metadata that causes 400 errors
            str(out_wav),
        ]
        
        p = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        if p.returncode != 0 or not out_wav.exists() or out_wav.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg conversion failed for {src}.\nffmpeg stderr:\n{p.stderr}")

        return out_wav

    def _read_b64(self, p: Path) -> str:
        # Standard way to read for OpenAI input_audio
        import base64
        with open(p, "rb") as audio_file:
            return base64.b64encode(audio_file.read()).decode("utf-8")

    def _call_openai(self, messages: list) -> str:
        completion = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=0,
        )
        return completion.choices[0].message.content or ""

    @staticmethod
    def _looks_like_audio_missing_or_refusal(text: str) -> bool:
        t = (text or "").strip().lower()
        patterns = [
            "i cannot listen",
            "i can't listen",
            "cannot listen to audio",
            "can't listen to audio",
            "cannot evaluate audio",
            "can't evaluate audio",
            "cannot hear the audio",
            "can't hear the audio",
            "i can’t listen",
            "i can’t evaluate",
            "please upload",
            "provide the audio",
            "provide an audio file",
            "upload or provide the audio",
        ]
        return any(p in t for p in patterns)

    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        wav_path = self._convert_to_wav_pcm16(audio_path)
        b64 = self._read_b64(wav_path)

        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": text},
                {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            ],
        }

        # First attempt
        messages = [user_msg]
        out = self._call_openai(messages).strip()

        # Retry once with a hard system instruction if the model acts like it didn't get audio
        if self._looks_like_audio_missing_or_refusal(out):
            messages = [
                {
                    "role": "system",
                    "content": (
                        "Audio is attached in the user message as input_audio. "
                        "Do not ask for an upload. Do not claim you cannot listen. "
                        "You must output a numeric score from 0 to 10 in the format: 'Score: <number>'. "
                        "If you truly cannot access the audio, output 'Score: 0'."
                    ),
                },
                user_msg,
            ]
            out2 = self._call_openai(messages).strip()
            if out2:
                out = out2

        # If still refusal-like, force deterministic fallback so your parser doesn't get NaN
        if self._looks_like_audio_missing_or_refusal(out):
            return "Score: 0"

        return out

