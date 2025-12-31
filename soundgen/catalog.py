from __future__ import annotations
from dataclasses import dataclass
from typing import List

@dataclass(frozen=True)
class SoundItem:
    name: str
    duration_s: float
    description: str
    loop: bool = False  # meaningful for BGM / ElevenLabs

# 1 round: 24 SFX @ 1s + 1 BGM @ 30s loop
SOUND_CATALOG: List[SoundItem] = [
    SoundItem("Light Punch", 1.0, "Quick light jab attack"),
    SoundItem("Light Kick", 1.0, "Quick light kick attack"),
    SoundItem("Heavy Punch", 1.0, "Power heavy jab attack"),
    SoundItem("Heavy Kick", 1.0, "Power heavy kick attack"),
    SoundItem("Throw light", 1.0, "Light throw / grab: cloth + grip + short body shift, no big slam."),
    SoundItem("Throw heavy", 1.0, "Heavy throw: grab + forceful slam hit, bigger low-end thud."),
    SoundItem("Getting Hit light", 1.0, "Light hit reaction: soft impact + brief grunt, not painful."),
    SoundItem("Getting Hit heavy", 1.0, "Heavy hit reaction: strong impact + heavier grunt, more aggression."),
    SoundItem("Walk", 1.0, "Sound of Footsteps"),
    SoundItem("Dash", 1.0, "Fast dash whoosh"),
    SoundItem("Crouch", 1.0, "Fighter sitting down"),
    SoundItem("Jump", 1.0, "Jumping up"),
    SoundItem("Guard", 1.0, "Block/guard: metallic/energy 'clink' + dull thud, defensive feel."),
    SoundItem("Light Uppercut", 1.0, "Uppercut light: + short tail."),
    SoundItem("Hard Uppercut", 1.0, "Hard uppercut: bigger rise + brief lightning crackle."),
    SoundItem("Dashkick", 1.0, "Dash kick: fast whoosh + kick hit, aggressive."),
    SoundItem("Jump Strike", 1.0, "jump and punch sound"),
    SoundItem("Light Fireball", 1.0, "Small energy projectile: charge blip + release, short airy trail."),
    SoundItem("Heavy Fireball", 1.0, "Large energy projectile: deeper charge + stronger release, longer trail."),
    SoundItem("Special Fireball", 1.0, "Signature projectile: layered charge + release + shimmer, heroic."),
    SoundItem("Landing", 1.0, "Landing thump: feet impact , short low thud."),
    SoundItem("EnergyChange", 1.0, "Energy shift: rising tone + pulse, conveys power-up."),
    SoundItem("Heartbeat", 1.0, "Heartbeat pulse: deep muffled thump, steady."),
    SoundItem("BorderAlert", 1.0, "UI danger alert: sharp beep/pulse + subtle rumble, urgent."),
    SoundItem("BGM", 30.0, "A fighting-game background music designed to loop seamlessly.", loop=True),
        SoundItem("fullfight", 40.0, "1 round of a fighting-game between two players.", loop=True),
]
