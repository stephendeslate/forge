"""Draft → Critique → Refine pipeline using two-model orchestration.

Fast model drafts and critiques; heavy model refines.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from enum import Enum

from forge.models.base import ModelBackend
from forge.prompts.refine import CRITIQUE_SYSTEM, DRAFT_SYSTEM, REFINE_SYSTEM


class PipelineStage(str, Enum):
    DRAFT = "draft"
    CRITIQUE = "critique"
    REFINE = "refine"
    DONE = "done"


@dataclass
class PipelineResult:
    draft: str = ""
    critique: str = ""
    refined: str = ""
    stage: PipelineStage = PipelineStage.DONE


OnStageCallback = Callable[[PipelineStage, str], Awaitable[None]]


class Pipeline:
    """Orchestrates draft → critique → refine across two models."""

    def __init__(
        self,
        drafter: ModelBackend,
        refiner: ModelBackend,
        critic: ModelBackend | None = None,
    ) -> None:
        self._drafter = drafter  # fast model
        self._refiner = refiner  # heavy model
        self._critic = critic or drafter  # critique model (defaults to drafter for backward compat)

    async def run(
        self,
        prompt: str,
        *,
        context: str = "",
        on_stage: OnStageCallback | None = None,
    ) -> PipelineResult:
        """Run the full pipeline and return all stages.

        Args:
            prompt: The user's request.
            context: Optional RAG context to inject.
            on_stage: Optional async callback(stage: PipelineStage, text: str) for progress.
        """
        result = PipelineResult()

        draft_system = DRAFT_SYSTEM
        if context:
            draft_system = f"{DRAFT_SYSTEM}\n\n{context}"

        # Stage 1: Draft (fast model)
        result.stage = PipelineStage.DRAFT
        if on_stage:
            await on_stage(PipelineStage.DRAFT, "")
        result.draft = await self._drafter.generate(prompt, system=draft_system)
        if on_stage:
            await on_stage(PipelineStage.DRAFT, result.draft)
        # Stage 2: Critique (configurable — defaults to drafter, can use heavy/Gemini)
        result.stage = PipelineStage.CRITIQUE
        if on_stage:
            await on_stage(PipelineStage.CRITIQUE, "")
        critique_prompt = (
            f"## Original Request\n{prompt}\n\n"
            f"## Draft Response\n{result.draft}"
        )
        result.critique = await self._critic.generate(
            critique_prompt, system=CRITIQUE_SYSTEM
        )
        if on_stage:
            await on_stage(PipelineStage.CRITIQUE, result.critique)
        # Stage 3: Refine (heavy model — produces final output)
        result.stage = PipelineStage.REFINE
        if on_stage:
            await on_stage(PipelineStage.REFINE, "")
        refine_prompt = (
            f"## Original Request\n{prompt}\n\n"
            f"## Draft Response\n{result.draft}\n\n"
            f"## Critique\n{result.critique}"
        )
        refine_system = REFINE_SYSTEM
        if context:
            refine_system = f"{REFINE_SYSTEM}\n\n{context}"
        result.refined = await self._refiner.generate(
            refine_prompt, system=refine_system
        )
        if on_stage:
            await on_stage(PipelineStage.REFINE, result.refined)
        result.stage = PipelineStage.DONE
        return result

    async def stream_refine(
        self,
        prompt: str,
        *,
        context: str = "",
    ) -> tuple[str, str, AsyncIterator[str]]:
        """Run draft + critique, then stream the refinement.

        Returns (draft, critique, refine_stream) so the caller can display
        the final output progressively while draft/critique run in the background.
        """
        draft_system = DRAFT_SYSTEM
        refine_system = REFINE_SYSTEM
        if context:
            draft_system = f"{DRAFT_SYSTEM}\n\n{context}"
            refine_system = f"{REFINE_SYSTEM}\n\n{context}"

        # Draft (fast)
        draft = await self._drafter.generate(prompt, system=draft_system)

        # Critique (configurable)
        critique_prompt = (
            f"## Original Request\n{prompt}\n\n"
            f"## Draft Response\n{draft}"
        )
        critique = await self._critic.generate(
            critique_prompt, system=CRITIQUE_SYSTEM
        )

        # Refine (heavy, streamed)
        refine_prompt = (
            f"## Original Request\n{prompt}\n\n"
            f"## Draft Response\n{draft}\n\n"
            f"## Critique\n{critique}"
        )
        stream = self._refiner.stream(refine_prompt, system=refine_system)

        return draft, critique, stream
