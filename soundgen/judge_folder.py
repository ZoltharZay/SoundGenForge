from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional, Any, List, Literal, Tuple

from tqdm import tqdm

# Load .env if available (non-fatal if not installed)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

# Package-relative imports (required for: python -m soundgen.judge_folder)
from .eval.flamingo_judge import AudioFlamingo3Judge  # IMPORTANT: you asked to include this

PromptStyle = Literal["guided", "penalty", "unguided"]
RunMode = Literal["one_step", "two_step"]

# --- Model Evaluator Configurations ---
# Keys must match what you pass to --model.
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # --- API-Based Models ---
    "gpt-4o": {
        "type": "api",
        "api_key": None,
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-audio-mini",  # IMPORTANT: actual OpenAI model name
    },

    # --- Local Models ---
    "kimi-audio": {
        "type": "local",
        "model_name": "moonshotai/Kimi-Audio-7B-Instruct",
        "trust_remote_code": True,
    },
    "qwen2-audio": {
        "type": "local",
        "model_name": "Qwen/Qwen2-Audio-7B-Instruct",
        "trust_remote_code": True,
    },
    "qwen3-omni": {
        "type": "local",
        "model_name": "Qwen/Qwen3-Omni-30B-A3B-Instruct",
        "trust_remote_code": True,
    },
    "omni-r1": {
        "type": "local",
        "model_name": "Haoz0206/Omni-R1",
        "trust_remote_code": True,
    },

    # --- Added: Audio Flamingo 3 ---
    "audio-flamingo-3": {
        "type": "local",
        "model_name": "nvidia/audio-flamingo-3-hf",
        "trust_remote_code": True,
    },
}


def _default_game_info() -> str:
    return "You are a Strict Audio Quality Evaluator. Your tone is clinical, critical, and unforgiving."


def _load_generation_prompt_map(prompt_json: Optional[Path]) -> Dict[str, str]:
    if prompt_json is None:
        return {}
    data = json.loads(prompt_json.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("prompt_json must be a JSON object mapping stem -> prompt string")
    return {str(k): str(v) for k, v in data.items()}


def _try_load_sound_catalog() -> Optional[Dict[str, Dict[str, str]]]:
    """
    Optional: uses your existing SOUND_CATALOG if present.
    Returns mapping: loose_key -> {name, description}
    """
    try:
        from .catalog import SOUND_CATALOG  # type: ignore
    except Exception:
        return None

    def loosen(x: str) -> str:
        return "".join(ch for ch in x.lower() if ch.isalnum())

    m: Dict[str, Dict[str, str]] = {}
    for item in SOUND_CATALOG:
        m[loosen(item.name)] = {
            "name": item.name,
            "description": getattr(item, "description", "") or "",
        }
    return m


def _with_iter_suffix(path: Path, iter_idx: int) -> Path:
    """
    Add __iterXX before extension (or at end if no extension).
    """
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    return path.with_name(f"{stem}__iter{iter_idx:02d}{suffix}")


def _discover_audio_jobs(audio_dir: Path) -> List[Tuple[str, Path]]:
    """
    If audio_dir points to a normal folder containing audio files, return one job: ("", audio_dir).
    If audio_dir ends with /all OR contains subfolders that each have an /audio folder, return jobs:
      [(subfolder_name, subfolder/audio), ...]
    """
    # If user explicitly passed a directory named "all", treat its parent as the root.
    if audio_dir.name == "all":
        root = audio_dir.parent
        jobs: List[Tuple[str, Path]] = []
        for sub in sorted(root.iterdir()):
            if not sub.is_dir():
                continue
            aud = sub / "audio"
            if aud.is_dir():
                jobs.append((sub.name, aud))
        return jobs

    # Otherwise, if this directory itself contains audio files, treat as single job
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    has_audio_files = any(p.is_file() and p.suffix.lower() in exts for p in audio_dir.iterdir())
    if has_audio_files:
        return [("", audio_dir)]

    # Otherwise, interpret it as a root folder where each subfolder has an /audio child
    jobs = []
    for sub in sorted(audio_dir.iterdir()):
        if not sub.is_dir():
            continue
        aud = sub / "audio"
        if aud.is_dir():
            jobs.append((sub.name, aud))
    return jobs


def _make_judge(
    *,
    model_key: str,
    prompt_style: PromptStyle,
    device: Optional[str],
    torch_dtype: Optional[str],
    max_new_tokens: int,
    trust_remote_code: bool,
    attn_implementation: str,
    use_fast_processor: bool,
    openai_api_key: Optional[str],
    openai_base_url: Optional[str],
    openai_model_name: Optional[str],
    openai_max_tokens: int,
):
    """
    Standalone loading via judge classes.

    model_key: your config key (e.g., 'qwen3-omni')
    actual HF/OpenAI model id: MODEL_CONFIGS[model_key]['model_name']
    """
    if model_key not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model key: {model_key}. Choices: {list(MODEL_CONFIGS.keys())}")

    cfg = MODEL_CONFIGS[model_key]
    actual_name = cfg["model_name"]

    if model_key == "gpt-4o":
        api_key = openai_api_key or os.environ.get("OPENAI_API_KEY") or cfg.get("api_key")
        if not api_key:
            raise ValueError("Missing OpenAI API key. Set OPENAI_API_KEY in env/.env or pass --openai-api-key.")

        return GPT4oJudge(
            api_key=api_key,
            base_url=openai_base_url or cfg.get("base_url"),
            model_name=openai_model_name or actual_name,  # defaults to gpt-audio-mini
            prompt_style=prompt_style,
            max_tokens=openai_max_tokens,
        )

    if model_key == "kimi-audio":
        return KimiAudioJudge(
            model_path=actual_name,
            prompt_style=prompt_style,
        )

    if model_key == "qwen2-audio":
        return Qwen2AudioJudge(
            model_id=actual_name,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            prompt_style=prompt_style,
            trust_remote_code=bool(cfg.get("trust_remote_code", trust_remote_code)),
        )

    if model_key == "qwen3-omni":
        return Qwen3OmniJudge(
            model_id=actual_name,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            prompt_style=prompt_style,
            trust_remote_code=bool(cfg.get("trust_remote_code", trust_remote_code)),
            attn_implementation=attn_implementation,
            use_fast_processor=use_fast_processor,
        )

    if model_key == "omni-r1":
        return OmniR1Judge(
            model_id=actual_name,
            device=device,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
            prompt_style=prompt_style,
            trust_remote_code=bool(cfg.get("trust_remote_code", trust_remote_code)),
            attn_implementation=attn_implementation,
        )

    if model_key == "audio-flamingo-3":
        return AudioFlamingo3Judge(
            model_id=actual_name,  # nvidia/audio-flamingo-3-hf
            device=device,
            torch_dtype=torch_dtype,
            prompt_style=prompt_style,
        )

    raise ValueError(f"Unhandled model key: {model_key}")


def _run_one(
    *,
    judge: Any,
    audio_dir: Path,
    out_json: Path,
    mode: RunMode,
    prompt_style: PromptStyle,
    game_info: str,
    prompt_map: Dict[str, str],
    catalog: Optional[Dict[str, Dict[str, str]]],
    model_key: str,
) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # Avoid reloading models; just update prompt_style between runs
    try:
        judge.prompt_style = prompt_style
    except Exception:
        pass

    def loosen(x: str) -> str:
        return "".join(ch for ch in x.lower() if ch.isalnum())

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    files = [p for p in sorted(audio_dir.iterdir()) if p.is_file() and p.suffix.lower() in exts]

    results: List[Dict[str, Any]] = []

    for audio_path in tqdm(files, desc=f"{model_key} | {audio_dir.name} | {mode} | {prompt_style}"):
        stem = audio_path.stem
        gen_prompt = prompt_map.get(stem, "")

        sfx_name = stem
        sfx_desc = ""
        if catalog:
            key = loosen(stem)
            if key in catalog:
                sfx_name = catalog[key]["name"]
                sfx_desc = catalog[key]["description"]

        audio_desc_turn = None
        audio_desc_text = None

        if mode == "two_step":
            audio_desc_turn = judge.describe_audio(audio_path=audio_path)
            audio_desc_text = audio_desc_turn.response

        pq_turn = judge.score_pq(
            audio_path=audio_path,
            game_info=game_info,
            sfx_name=sfx_name,
            sfx_description=sfx_desc,
            generation_prompt=gen_prompt,
            audio_description=audio_desc_text,
        )
        ce_turn = judge.score_ce(
            audio_path=audio_path,
            game_info=game_info,
            sfx_name=sfx_name,
            sfx_description=sfx_desc,
            generation_prompt=gen_prompt,
            audio_description=audio_desc_text,
        )
        ci_turn = judge.score_ci(
            audio_path=audio_path,
            game_info=game_info,
            sfx_name=sfx_name,
            sfx_description=sfx_desc,
            generation_prompt=gen_prompt,
            audio_description=audio_desc_text,
        )

        results.append(
            {
                "file": str(audio_path),
                "model_key": model_key,
                "model_name": MODEL_CONFIGS[model_key]["model_name"],
                "mode": mode,
                "prompt_style": prompt_style,
                "game_info": game_info,
                "sfx_name": sfx_name,
                "sfx_description": sfx_desc,
                "generation_prompt": gen_prompt,
                "describe": (
                    {"prompt": audio_desc_turn.prompt, "response": audio_desc_turn.response}
                    if audio_desc_turn is not None
                    else None
                ),
                "pq": {"prompt": pq_turn.prompt, "response": pq_turn.response, "score": pq_turn.score},
                "ce": {"prompt": ce_turn.prompt, "response": ce_turn.response, "score": ce_turn.score},
                "ci": {"prompt": ci_turn.prompt, "response": ci_turn.response, "score": ci_turn.score},
            }
        )

        # incremental write
        out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    out_json.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--audio-dir",
        required=True,
        help=(
            "Folder containing audio files OR a root folder whose subfolders each contain an 'audio' folder. "
            "If you pass a path ending with '/all', it will treat the parent as the root and evaluate all subfolders."
        ),
    )

    # Single-run output (or multi-iter output with suffixes)
    ap.add_argument("--out-json", default=None, help="Single-run output JSON path (or base name if --num-iters>1).")

    # Sweep output (6 runs; with iterations it becomes 6*num-iters)
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "If set, runs all (one_step/two_step) × (guided/penalty/unguided) and writes JSON files here. "
            "If --audio-dir points to a root (or ends with /all), results will be written into subfolders in out-dir "
            "matching each discovered input subfolder name."
        ),
    )

    ap.add_argument("--num-iters", type=int, default=1, help="Repeat each run N times and save each iteration separately.")

    ap.add_argument("--model", required=True, choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--mode", choices=["one_step", "two_step"], default="one_step")
    ap.add_argument("--prompt-style", choices=["guided", "penalty", "unguided"], default="guided")
    ap.add_argument("--game-info", default=_default_game_info())
    ap.add_argument("--prompt-json", default=None, help="Optional JSON mapping stem->generation prompt used.")

    # Local/HF knobs (used by qwen/omni/flamingo judges)
    ap.add_argument("--device", default=None, help="cuda or cpu (default auto).")
    ap.add_argument("--torch-dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--trust-remote-code", action="store_true")
    ap.add_argument("--attn-implementation", default="flash_attention_2")
    ap.add_argument("--use-fast-processor", action="store_true")

    # OpenAI knobs
    ap.add_argument("--openai-api-key", default=None)
    ap.add_argument("--openai-base-url", default=None)
    ap.add_argument("--openai-model-name", default=None)  # override config model_name
    ap.add_argument("--openai-max-tokens", type=int, default=256)

    args = ap.parse_args()

    if args.num_iters < 1:
        raise ValueError("--num-iters must be >= 1")

    audio_dir = Path(args.audio_dir)
    if not audio_dir.exists():
        raise FileNotFoundError(f"--audio-dir not found: {audio_dir}")

    prompt_map = _load_generation_prompt_map(Path(args.prompt_json)) if args.prompt_json else {}
    catalog = _try_load_sound_catalog()

    # Discover jobs: either one folder, or many subfolders each with /audio
    jobs = _discover_audio_jobs(audio_dir)
    if not jobs:
        raise ValueError(
            f"No audio jobs found under --audio-dir={audio_dir}. "
            "Expected either audio files directly, or subfolders each containing an 'audio' folder."
        )

    # Instantiate judge ONCE (so local models are loaded once).
    judge = _make_judge(
        model_key=args.model,
        prompt_style=args.prompt_style,
        device=args.device,
        torch_dtype=args.torch_dtype,
        max_new_tokens=args.max_new_tokens,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        use_fast_processor=args.use_fast_processor,
        openai_api_key=args.openai_api_key,
        openai_base_url=args.openai_base_url,
        openai_model_name=args.openai_model_name,
        openai_max_tokens=args.openai_max_tokens,
    )

    # Sweep mode: all combos
    if args.out_dir:
        out_root = Path(args.out_dir)
        out_root.mkdir(parents=True, exist_ok=True)

        modes: List[RunMode] = ["one_step", "two_step"]
        styles: List[PromptStyle] = ["guided", "penalty", "unguided"]

        for job_name, job_audio_dir in jobs:
            # If single-folder mode, job_name will be ""
            out_dir = out_root if job_name == "" else (out_root / job_name)
            out_dir.mkdir(parents=True, exist_ok=True)

            for mode in modes:
                for style in styles:
                    base = out_dir / f"{args.model}__{mode}__{style}.json"
                    for it in range(1, args.num_iters + 1):
                        out_json = base if args.num_iters == 1 else _with_iter_suffix(base, it)
                        _run_one(
                            judge=judge,
                            audio_dir=job_audio_dir,
                            out_json=out_json,
                            mode=mode,
                            prompt_style=style,
                            game_info=args.game_info,
                            prompt_map=prompt_map,
                            catalog=catalog,
                            model_key=args.model,
                        )
        return

    # Single-run mode (possibly repeated with suffixes)
    if not args.out_json:
        raise ValueError("Provide either --out-json (single run) or --out-dir (sweep).")

    # If multiple jobs were discovered, interpret out_json as an output root directory-like base name.
    base_out = Path(args.out_json)
    base_out.parent.mkdir(parents=True, exist_ok=True)

    if len(jobs) > 1:
        # Treat base_out as a folder path; write per-job files inside it
        out_root = base_out
        out_root.mkdir(parents=True, exist_ok=True)

        for job_name, job_audio_dir in jobs:
            out_dir = out_root / job_name
            out_dir.mkdir(parents=True, exist_ok=True)

            base = out_dir / f"{args.model}__{args.mode}__{args.prompt_style}.json"
            for it in range(1, args.num_iters + 1):
                out_json = base if args.num_iters == 1 else _with_iter_suffix(base, it)
                _run_one(
                    judge=judge,
                    audio_dir=job_audio_dir,
                    out_json=out_json,
                    mode=args.mode,
                    prompt_style=args.prompt_style,
                    game_info=args.game_info,
                    prompt_map=prompt_map,
                    catalog=catalog,
                    model_key=args.model,
                )
        return

    # Normal single job case
    _, job_audio_dir = jobs[0]
    for it in range(1, args.num_iters + 1):
        out_json = base_out if args.num_iters == 1 else _with_iter_suffix(base_out, it)
        _run_one(
            judge=judge,
            audio_dir=job_audio_dir,
            out_json=out_json,
            mode=args.mode,
            prompt_style=args.prompt_style,
            game_info=args.game_info,
            prompt_map=prompt_map,
            catalog=catalog,
            model_key=args.model,
        )


if __name__ == "__main__":
    main()

