from __future__ import annotations
import os, re
from pathlib import Path

def slugify(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s

def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p

def target_audio_path(out_dir: str | Path, item_name: str, ext: str) -> Path:
    return Path(out_dir) / f"{slugify(item_name)}.{ext.lstrip('.')}"
