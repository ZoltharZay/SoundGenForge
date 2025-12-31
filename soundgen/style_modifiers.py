from __future__ import annotations

STYLE_MODIFIERS = {
    # Impacts
    "light punch": "tight transient; bright snap; short decay; minimal low-end",
    "heavy punch": "heavier low-mid thump; crisp attack; controlled sub; short tail",
    "heavy kick": "deep body thud; strong whoosh lead-in; punchy transient",
    "heavy kick 2": "aggressive whoosh; harder slam; slightly wider stereo",

    # Throws / hits
    "throw light": "cloth grab; quick movement; light slam punctuation",
    "throw heavy": "strong grab; heavier slam; thick low-end",
    "getting hit light": "soft impact; restrained pain; quick decay",
    "getting hit heavy": "hard impact; more weight; slightly longer decay",

    # Movement
    "walk": "loop-friendly; soft steps; no sharp spikes",
    "dash": "fast whoosh; friction hiss; abrupt stop",
    "crouch": "cloth rustle; subtle slide; low intensity",
    "jump": "springy whoosh; quick lift-off",
    "landing": "short low thud; dust/cloth; no boom",

    # Defense
    "guard": "solid parry click; muted thud; defensive tone",

    # Specials
    "light uppercut": "rising whoosh; clean snap; short tail",
    "hard uppercut with lightning": "big impact; brief electric crackle; bright spark detail",
    "dashkick": "long whoosh; strong hit; aggressive",
    "jump strike": "air whoosh; bright hit; minimal tail",

    # Projectiles
    "light fireball": "compact energy blip; airy fizz; short trail",
    "heavy fireball": "stronger charge; heavier release; thicker low-end",
    "special fireball": "layered charge; heroic shimmer; clean release zap",

    # UI / biosignals
    "energychange": "smooth rise; pulse; sci-fi clean tone",
    "heartbeat": "deep clean thump; steady; minimal harmonics",
    "borderalert": "urgent beep/pulse; subtle rumble; clear attention",

    # BGM
    "bgm": "mid-tempo groove; tight drums; controlled bass; minimal intro/outro; loop-safe ending",
}

def get_style_modifiers(name: str) -> str:
    return STYLE_MODIFIERS.get(name.strip().lower(), "none")

