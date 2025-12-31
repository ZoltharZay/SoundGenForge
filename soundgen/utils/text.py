from __future__ import annotations
import re

MAX_ELEVENLABS_CHARS = 450

def shorten_for_elevenlabs(text: str, max_chars: int = MAX_ELEVENLABS_CHARS) -> str:
    """
    ElevenLabs sound-generation text has a hard length cap (450 chars).
    This function keeps the key semantics but compresses aggressively.
    """
    t = " ".join(text.strip().split())  # collapse whitespace

    # Remove verbose boilerplate that doesn't help generation
    drop_phrases = [
        "You are designing audio for a 2D fighting game.",
        "Create a single, clean, professional",
        "Style constraints:",
        "NEGATIVE CONSTRAINTS (avoid all of these):",
        "Deliver one cohesive sound only.",
        "Constraints:",
    ]
    for p in drop_phrases:
        t = t.replace(p, "")

    # Compact separators
    t = t.replace("Sound effect name:", "SFX:")
    t = t.replace("Description:", "Desc:")
    t = t.replace("Style modifiers:", "Mods:")
    t = re.sub(r"\s*;\s*", "; ", t)
    t = re.sub(r"\s*,\s*", ", ", t)
    t = re.sub(r"\s+", " ", t).strip()

    if len(t) <= max_chars:
        return t

    # Hard truncate while trying to avoid cutting mid-word
    truncated = t[: max_chars - 1]
    truncated = truncated.rsplit(" ", 1)[0]
    return truncated

