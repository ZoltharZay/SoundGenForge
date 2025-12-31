from __future__ import annotations
import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

from .pipeline import PipelineConfig, run_generation_round
from .generators.tangoflux import TangoFluxGenerator
from .generators.audiox import AudioXGenerator
from .generators.elevenlabs_sfx import ElevenLabsGenerator

def build_generator(name: str):
    name = name.lower().strip()
    if name == "tangoflux":
        return TangoFluxGenerator()
    if name == "audiox":
        return AudioXGenerator()
    if name == "elevenlabs":
        api_key = os.environ.get("ELEVENLABS_API_KEY", "").strip()
        if not api_key:
            raise SystemExit("Missing ELEVENLABS_API_KEY. Set it in your environment or .env file.")
        return ElevenLabsGenerator(api_key=api_key)
    raise SystemExit(f"Unknown model: {name}. Use one of: tangoflux, audiox, elevenlabs")

def main():
    load_dotenv()

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["tangoflux", "audiox", "elevenlabs"])
    ap.add_argument("--method", required=True, choices=["one_step", "two_step"])
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--max-attempts", type=int, default=5)
    ap.add_argument("--ce-threshold", type=float, default=6.0)
    ap.add_argument("--pq-threshold", type=float, default=6.0)
    ap.add_argument(
    "--evaluators",
    type=str,
    default="None",
    help="Comma-separated evaluators: audiobox,flamingo3 (e.g., 'audiobox' or 'flamingo3' or 'audiobox,flamingo3')",
)
    ap.add_argument(
        "--flamingo-threshold",
        type=float,
        default=6.0,
        help="Threshold (0-10) for Audio Flamingo 3 evaluator acceptance.",
    )
    ap.add_argument(
        "--use-feedback",
        action="store_true",
        help="If set, append a short feedback sentence from evaluator(s) to the next retry prompt.",
    )

    args = ap.parse_args()
    evaluators = tuple(
        x.strip().lower()
        for x in args.evaluators.split(",")
        if x.strip()
    )

    gen = build_generator(args.model)

    cfg = PipelineConfig(
        method=args.method,
        out_dir=Path(args.out),
        max_attempts=args.max_attempts,
        pq_threshold=args.pq_threshold,
        flamingo_threshold=args.flamingo_threshold,
        evaluators=evaluators,
        use_feedback=bool(args.use_feedback),
    )


    run_generation_round(generator=gen, cfg=cfg)

if __name__ == "__main__":
    main()
