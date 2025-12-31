from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional
import csv
import math

from tqdm import tqdm

from .utils.audio_io import ffmpeg_to_wav
from .style_modifiers import get_style_modifiers
from .catalog import SOUND_CATALOG
from .prompts import render_prompt
from .utils.paths import ensure_dir, target_audio_path

from .eval.audiobox import AudioboxAestheticsScorer


@dataclass(frozen=True)
class PipelineConfig:
    method: str  # one_step or two_step
    out_dir: Path
    max_attempts: int = 5
    _attempt_counter: int = 0

    # thresholds
    pq_threshold: float = 0.6
    flamingo_threshold: float = 6.0

    # evaluator modes: ("audiobox",) or ("flamingo3",) or ("audiobox","flamingo3")
    evaluators: Tuple[str, ...] = ("audiobox",)

    # append 1 short feedback sentence for next attempt (if enabled)
    use_feedback: bool = True

    keep_failed_attempts: bool = True


def run_generation_round(*, generator, cfg: PipelineConfig) -> None:
    out_dir = ensure_dir(cfg.out_dir)
    audio_dir = ensure_dir(out_dir / "audio")
    attempts_dir = ensure_dir(out_dir / "attempts")
    scores_csv = out_dir / "scores.csv"

    # Instantiate enabled evaluators
    audiobox = None
    flamingo = None

    if cfg.method == "two_step":
        if "audiobox" in cfg.evaluators:
            audiobox = AudioboxAestheticsScorer()
        if "flamingo3" in cfg.evaluators:
            from .eval.flamingo3 import AudioFlamingo3Scorer  # lazy import
            flamingo = AudioFlamingo3Scorer()

    with open(scores_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "item_name", "attempt", "prompt", "path",
                "ce", "pq",
                "flamingo_score_0_10", "flamingo_feedback", "flamingo_raw",
                "accepted",
            ],
        )
        writer.writeheader()

        for item in tqdm(SOUND_CATALOG, desc=f"Generating ({generator.name})"):
            is_bgm = item.name.strip().lower() == "bgm"
            mods = get_style_modifiers(item.name)

            base_prompt = render_prompt(
                item.name, item.duration_s, item.description,
                is_bgm=is_bgm,
                style_modifiers=mods,
                include_negative=True,
            ).text

            # Decide file extension by generator
            ext = "wav" if generator.name in {"tangoflux", "audiox"} else "mp3"
            final_path = target_audio_path(audio_dir, item.name, ext)

            # ONE STEP
            if cfg.method == "one_step":
                generator.generate(
                    prompt=base_prompt,
                    duration_s=item.duration_s,
                    out_path=final_path,
                    loop=item.loop,
                    seed=None,
                )
                writer.writerow({
                    "item_name": item.name,
                    "attempt": 1,
                    "prompt": base_prompt,
                    "path": str(final_path),
                    "ce": "",
                    "pq": "",
                    "flamingo_score_0_10": "",
                    "flamingo_feedback": "",
                    "accepted": True,
                })
                f.flush()
                continue

            # TWO STEP (retry)
            current_prompt = base_prompt
            accepted = False
            best = None  # (best_score, tmp_path, ce, pq, fl_score, prompt)

            for attempt in range(1, cfg.max_attempts + 1):
                tmp_path = target_audio_path(attempts_dir, f"{item.name}__attempt_{attempt}", ext)

                generator.generate(
                    prompt=current_prompt,
                    duration_s=item.duration_s,
                    out_path=tmp_path,
                    loop=item.loop,
                    seed=None,
                )

                # Prefer WAV for evaluation (especially audiobox). :contentReference[oaicite:2]{index=2}
                eval_path = tmp_path
                if tmp_path.suffix.lower() == ".mp3":
                    wav_path = tmp_path.with_suffix(".wav")
                    try:
                        ffmpeg_to_wav(tmp_path, wav_path, sample_rate=44100)
                        eval_path = wav_path
                    except Exception:
                        eval_path = tmp_path  # fallback

                ce = pq = float("nan")
                fl_score = float("nan")
                fl_fb = ""
                fl_raw = ""   

                # Audiobox CE/PQ
                if audiobox is not None:
                    try:
                        ab = audiobox.score_file(eval_path)
                        ce, pq = ab.ce, ab.pq
                    except Exception:
                        ce, pq = float("nan"), float("nan")

                # Flamingo score + feedback (0-10)
                if flamingo is not None:
                    try:
                        fl = flamingo.score_file(eval_path, generation_prompt=current_prompt)
                        fl_score, fl_fb, fl_raw = fl.score_0_10, fl.feedback, fl.raw
                    except Exception:
                        fl_score, fl_fb, fl_raw = float("nan"), "", ""

                # Acceptance rule depends on enabled evaluators
                ok = True
                if audiobox is not None:
                    pq_normalized = pq / 10.0
                    ok = ok and (pq_normalized >= cfg.pq_threshold)
                if flamingo is not None:
                    ok = ok and (fl_score >= cfg.flamingo_threshold)

                writer.writerow({
                    "item_name": item.name,
                    "attempt": attempt,
                    "prompt": current_prompt,
                    "path": str(tmp_path),
                    "ce": ce,
                    "pq": pq,
                    "flamingo_score_0_10": fl_score,
                    "flamingo_feedback": fl_fb,
                    "flamingo_raw": fl_raw,
                    "accepted": ok,
                })
                f.flush()

                # Track "best" attempt for fallback
                def _nz(x: float) -> float:
                    return 0.0 if (x != x) else float(x)  # NaN -> 0

                composite = _nz(pq) + _nz(ce) + _nz(fl_score)  # simple tie-breaker
                if best is None or composite > best[0]:
                    best = (composite, tmp_path, ce, pq, fl_score, current_prompt)

                if ok:
                    tmp_path.replace(final_path)
                    accepted = True
                    break

                # Add 1-sentence feedback for next attempt (optional)
                if cfg.use_feedback:
                    fb = (fl_fb or "").strip()
                    if not fb:
                        pq_variations = [
            "Reduce noise and reverb; make the transient cleaner and louder.",
            "Minimize background hiss and sharpen the audio",
            "Focus on a drier recording with a punchier, cleaner transient.",
            "Ensure the audio is clearer; boost the impact and cut the reverb.",
            "Aim for a cleaner signal-to-noise ratio and a much sharper start.",
               "Aim for a cleaner signal-to-noise ratio and Reduce noise and reverb"
        ]
                        feedback = pq_variations[attempt]
                    current_prompt = (base_prompt + " " + feedback).strip()

            # If never accepted, keep best attempt as final (with suffix)
            if (not accepted) and best is not None:
                _, best_path, ce, pq, fl_score, _ = best
                # Name includes key scores to help later filtering
                final_fallback = target_audio_path(
                    audio_dir,
                    f"{item.name}__BEST_FAIL_PQ{pq:.2f}_CE{ce:.2f}_FL{fl_score:.2f}",
                    ext
                )
                best_path.replace(final_fallback)

