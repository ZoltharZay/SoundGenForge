from __future__ import annotations
from dataclasses import dataclass

# Single shared SFX template with placeholders:
#   {sfx_name}, {duration_s}, {description}, 
PROMPT_TEMPLATE = (
    "Generate a professional {duration_s:.1f}-second sound effect for a 2D fighting game. "
    "Sound effect name: {sfx_name}. "
    "Description: {description} "
)

# BGM prompt (loopable)
BGM_PROMPT_TEMPLATE = (
    "Compose a {duration_s:.0f}-second looping background music track for a 2D fighting game. "
    "Genre: modern arcade/fighting game BGM "
    "Constraints: seamless loop (no abrupt start/end), consistent loudness, clean mix,"
)
# Optional: appended only on retries
FEEDBACK_APPEND_TEMPLATE = " Improve: {feedback}"


@dataclass(frozen=True)
class RenderedPrompt:
    text: str

def render_prompt(
    sfx_name: str,
    duration_s: float,
    description: str,
    is_bgm: bool,
    style_modifiers: str = "none",
    include_negative: bool = True,
) -> RenderedPrompt:
    style_modifiers = (style_modifiers or "none").strip()

    if is_bgm:
        base = BGM_PROMPT_TEMPLATE.format(duration_s=duration_s, style_modifiers=style_modifiers)
    else:
        base = PROMPT_TEMPLATE.format(
            sfx_name=sfx_name,
            duration_s=duration_s,
            description=description,
            style_modifiers=style_modifiers,
        )


    return RenderedPrompt(base)

