from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import abc
import re


PromptStyle = Literal["guided", "penalty", "unguided"]


@dataclass(frozen=True)
class JudgeTurn:
    prompt: str
    response: str
    score: float  # parsed best-effort; NaN if not found


def _parse_score(text: str) -> float:
    """
    Best-effort parsing for:
      Score: 7
      score - 7/10
      Score=7.5
    Returns NaN if no number found.
    """
    m = re.search(r"\bScore\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))

    m = re.search(r"\b([0-9]+(?:\.[0-9]+)?)\s*/\s*10\b", text)
    if m:
        return float(m.group(1))

    m = re.search(r"\bscore\b.*?\b([0-9]+(?:\.[0-9]+)?)\b", text, flags=re.IGNORECASE | re.DOTALL)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass

    return float("nan")


class BasePromptedAudioJudge(abc.ABC):
    """
    Common prompting + scoring logic shared by all backends.

    Backends must implement:
      - _ask(text=..., audio_path=..., max_new_tokens=...)

    Runner controls one_step vs two_step:
      - in two_step: runner calls describe_audio() first, then passes audio_description into score_*()
    """

    def __init__(self, *, prompt_style: PromptStyle = "guided") -> None:
        self.prompt_style: PromptStyle = prompt_style

    @abc.abstractmethod
    def _ask(self, *, text: str, audio_path: Path, max_new_tokens: int = 1000) -> str:
        raise NotImplementedError

    # -----------------------------
    # Prompt style builders
    # -----------------------------
    def _style_guided(self, *, header: str, task: str, criteria: str, rubric: str) -> str:
        return (
            f"{header}\n\n"
            f"{task}\n"
            f"**Criteria:** {criteria}\n\n"
            f"**Scoring Guidelines:**\n{rubric}\n\n"
            f"You are a Strict Audio Quality Evaluator. Your tone is clinical, critical, and unforgiving. "
            f"Your goal is to find reasons to LOWER the score based on the criteria.\n"
            f"**Constraint:** Output a score from 0-10.\n"
            f"Score: "
        )

    def _style_penalty(self, *, header: str, task: str, criteria: str, penalty_rubric: str) -> str:
        return (
            f"{header}\n\n"
            f"{task}\n"
            f"**Criteria:** {criteria}\n\n"
            f"**Penalty Scoring:**\n"
            f"(Start from 10. Subtract penalties. Clamp final score to 0-10.)\n"
            f"{penalty_rubric}\n\n"
            f"You are a Strict Audio Quality Evaluator. Your tone is clinical, critical, and unforgiving. "
            f"Your goal is to find reasons to LOWER the score based on the criteria.\n"
            f"**Constraint:** Output a score from 0-10.\n"
            f"Score: "
        )

    def _style_unguided(self, *, header: str, task: str) -> str:
        return (
            f"{header}\n\n"
            f"{task}\n"
            f"You are a Strict Audio Quality Evaluator. Your tone is clinical, critical, and unforgiving.\n"
            f"**Constraint:** Output a score from 0-10.\n"
            f"Score: "
        )

    def _build_prompt(
        self,
        *,
        game_info: str,
        sfx_name: str,
        sfx_description: str,
        audio_description: Optional[str],
        section_title: str,
        task_line: str,
        criteria_line: str,
        guided_rubric: str,
        penalty_rubric: str,
    ) -> str:
        header = (
            f"{game_info}\n"
            f"Sound effect name: {sfx_name}\n"
            f"TARGET SOUND DESCRIPTION (GROUND TRUTH): {sfx_description}\n"
            + (f"OBSERVED AUDIO DESCRIPTION: {audio_description}\n" if audio_description else "")
            + "\n"
            + f"{section_title}\n"
        )

        compare_instruction = ""
        if audio_description:
            compare_instruction = (
                "\n**Two-step requirement:** First compare OBSERVED AUDIO DESCRIPTION vs TARGET SOUND DESCRIPTION.\n"
                "- Identify mismatches (missing elements, wrong identity, wrong intensity, wrong timing).\n"
                "- Use those mismatches as negative evidence when scoring.\n"
            )

        task_with_compare = task_line + compare_instruction

        if self.prompt_style == "guided":
            return self._style_guided(
                header=header,
                task=task_with_compare,
                criteria=criteria_line,
                rubric=guided_rubric,
            )
        if self.prompt_style == "penalty":
            return self._style_penalty(
                header=header,
                task=task_with_compare,
                criteria=criteria_line,
                penalty_rubric=penalty_rubric,
            )
        return self._style_unguided(header=header, task=task_with_compare)

    # -----------------------------
    # Public API
    # -----------------------------
    def describe_audio(self, *, audio_path: Path) -> JudgeTurn:
        prompt = (
            "Task: Describe what this audio sounds like in detail using 2-3 sentences.\n"
            "Output format:\n"
            "Description: <2-3 sentences>"
        )
        resp = self._ask(text=prompt, audio_path=audio_path, max_new_tokens=180)
        return JudgeTurn(prompt=prompt, response=resp, score=float("nan"))

    def score_pq(
        self,
        *,
        audio_path: Path,
        game_info: str,
        sfx_name: str,
        sfx_description: str,
        generation_prompt: str,  # kept for API compatibility
        audio_description: Optional[str] = None,
    ) -> JudgeTurn:
        guided_rubric = (
            "  - **9-10 :** Perfect adherence; the technical fidelity matches every word of the ground truth.\n"
            "  - **7-8 :** Strong match; clear textures that align well with the ground truth.\n"
            "  - **5-6 :** Vague match; functional but lacks the specific technical detail of the ground truth.\n"
            "  - **3-4 :** Poor match; technical errors or textures contradict the ground truth.\n"
            "  - **0-2 :** Complete failure; technically unrecognizable compared to the ground truth."
        )

        penalty_rubric = (
            "  - Wrong physical texture/material vs ground truth: -10 points\n"
            "  - Audible clipping / harsh distortion / broken rendering: -8 points\n"
            "  - Unwanted noise/artifacts (hiss, crackles, buzzing, bad edit points): -6 points\n"
            "  - Tonal balance/dynamics mismatch (too muffled/boomy/flat/overcompressed) vs request: -4 points\n"
            "  - Minor technical imperfections (small artifact, slight EQ mismatch): -2 points"
        )

        prompt = self._build_prompt(
            game_info=game_info,
            sfx_name=sfx_name,
            sfx_description=sfx_description,
            audio_description=audio_description,
            section_title="Production Quality (PQ) Assessment",
            task_line="**Task:** Evaluate technical fidelity relative to the ground truth.",
            criteria_line="Does the audio accurately reproduce the physical textures requested?",
            guided_rubric=guided_rubric,
            penalty_rubric=penalty_rubric,
        )

        resp = self._ask(text=prompt, audio_path=audio_path)
        return JudgeTurn(prompt=prompt, response=resp, score=_parse_score(resp))

    def score_ce(
        self,
        *,
        audio_path: Path,
        game_info: str,
        sfx_name: str,
        sfx_description: str,
        generation_prompt: str,
        audio_description: Optional[str] = None,
    ) -> JudgeTurn:
        guided_rubric = (
            "  - **9-10 :** Aesthetic perfection; evokes the exact 'hype' or energy requested in the ground truth.\n"
            "  - **7-8 :** High impact; very satisfying and aligns with the energy of the ground truth.\n"
            "  - **5-6 :** Generic impact; functional but lacks the specific 'juice' requested.\n"
            "  - **3-4 :** Weak/Mismatched; energy significantly lower or different than the ground truth.\n"
            "  - **0-2 :** Jarring; fails to create the enjoyment or aesthetic intended by the ground truth."
        )

        penalty_rubric = (
            "  - Energy/hype clearly below or opposite of ground truth (e.g., 'explosive' but weak): -10 points\n"
            "  - Aesthetic/timbre mismatch vs description (cartoony vs realistic, harsh vs smooth, etc.): -8 points\n"
            "  - Unsatisfying impact/envelope relative to request (no punch, weak transient, wrong tail): -6 points\n"
            "  - Generic/stock feel with little character compared to description: -4 points\n"
            "  - Minor taste issues (slight dullness/overbrightness): -2 points"
        )

        prompt = self._build_prompt(
            game_info=game_info,
            sfx_name=sfx_name,
            sfx_description=sfx_description,
            audio_description=audio_description,
            section_title="Content Enjoyment (CE) Assessment",
            task_line="**Task:** Evaluate the aesthetic impact/enjoyment relative to the ground truth.",
            criteria_line="Does the sound deliver the energy level described (e.g., 'explosive' should feel powerful)?",
            guided_rubric=guided_rubric,
            penalty_rubric=penalty_rubric,
        )

        resp = self._ask(text=prompt, audio_path=audio_path)
        return JudgeTurn(prompt=prompt, response=resp, score=_parse_score(resp))

    def score_ci(
        self,
        *,
        audio_path: Path,
        game_info: str,
        sfx_name: str,
        sfx_description: str,
        generation_prompt: str,
        audio_description: Optional[str] = None,
    ) -> JudgeTurn:
        guided_rubric = (
            "  - **9-10 :** Instantly recognizable; perfect gameplay feedback for the action in the ground truth.\n"
            "  - **7-8 :** High readability; the identity of the move in the ground truth is clear.\n"
            "  - **5-6 :** Ambiguous; could be the action, but it is not distinct.\n"
            "  - **3-4 :** Confusing; suggests a different action than the ground truth.\n"
            "  - **0-2 :** Misleading; provides incorrect feedback for the action described in the ground truth."
        )

        penalty_rubric = (
            "  - Sound communicates incorrect information (contradicts described properties): -10 points\n"
            "  - Sound lacks defining characteristics (misses critical aspects from description): -8 points\n"
            "  - Sound is non-specific/too generic (could represent multiple different actions): -6 points\n"
            "  - Unclear temporal features (attack/transients don't match described intensity/timing): -4 points\n"
            "  - Minor mismatches (partially represents description with slight inconsistencies): -2 points"
        )

        prompt = self._build_prompt(
            game_info=game_info,
            sfx_name=sfx_name,
            sfx_description=sfx_description,
            audio_description=audio_description,
            section_title="Content Informativeness (CI) Assessment",
            task_line="**Task:** Evaluate the clarity of information. Does the sound communicate the specific action described in the ground truth.",
            criteria_line="If the audio matches the ground truth described above or does it sound like a different move?",
            guided_rubric=guided_rubric,
            penalty_rubric=penalty_rubric,
        )

        resp = self._ask(text=prompt, audio_path=audio_path)
        return JudgeTurn(prompt=prompt, response=resp, score=_parse_score(resp))

