"""Exemplar learning — capture cloud model successes for local model improvement.

When cloud models (Gemini) succeed at tasks the local model struggled with,
we store the (task, solution) pair as an exemplar. On future similar tasks,
relevant exemplars are retrieved and injected as few-shot context into the
local model's system prompt.

Over time, this creates a feedback loop: the local model gets better prompts
because it has examples of how cloud models solved similar problems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.config import settings
from forge.log import get_logger
from forge.models.embeddings import embed_single, format_embedding_for_pg

if TYPE_CHECKING:
    from forge.storage.database import Database, ExemplarRow

logger = get_logger(__name__)


async def capture_exemplar(
    db: Database,
    project: str,
    task_type: str,
    task_description: str,
    solution_text: str,
    model_source: str,
    outcome_score: float = 0.5,
) -> int | None:
    """Embed and store a cloud model success as an exemplar.

    Args:
        db: Database connection.
        project: Project identifier.
        task_type: One of 'recovery', 'planning', 'critique'.
        task_description: The user's original request or problem description.
        solution_text: The cloud model's full response/solution.
        model_source: Which model produced this (e.g. "gemini-primary").
        outcome_score: Initial confidence (0.0-1.0). Updated later by outcome tracking.

    Returns:
        Exemplar ID, or None if capture failed.
    """
    if not settings.agent.exemplar_enabled:
        return None

    # Truncate to keep embeddings meaningful (not dominated by boilerplate)
    desc_truncated = task_description[:2000]
    solution_truncated = solution_text[:4000]
    embed_text = f"{desc_truncated}\n\n{solution_truncated}"

    try:
        embedding = await embed_single(embed_text)
        embedding_str = format_embedding_for_pg(embedding)

        exemplar_id = await db.save_exemplar(
            project, task_type, desc_truncated, solution_truncated,
            outcome_score, model_source, embedding_str,
        )

        # Auto-prune if over threshold
        max_exemplars = settings.agent.exemplar_max_per_project
        count = await db.count_exemplars(project)
        if count > max_exemplars:
            pruned = await db.prune_exemplars(project, keep=max_exemplars)
            if pruned > 0:
                logger.debug("Pruned %d exemplars for project %s", pruned, project)

        logger.debug(
            "Captured %s exemplar #%d from %s (score=%.1f)",
            task_type, exemplar_id, model_source, outcome_score,
        )
        return exemplar_id

    except Exception:
        logger.debug("Failed to capture exemplar", exc_info=True)
        return None


async def retrieve_exemplars(
    db: Database,
    project: str,
    task_description: str,
    *,
    task_type: str | None = None,
    limit: int | None = None,
    min_score: float | None = None,
) -> tuple[str, list[int]]:
    """Retrieve similar past solutions and format as few-shot context.

    Returns (prompt_section, exemplar_ids) — the prompt section is ready for
    injection, and the IDs are for outcome tracking. Returns ("", []) if
    no relevant exemplars found or exemplars are disabled.
    """
    if not settings.agent.exemplar_enabled or not task_description.strip():
        return "", []

    effective_limit = limit if limit is not None else settings.agent.exemplar_max_inject
    effective_min_score = min_score if min_score is not None else settings.agent.exemplar_min_score

    try:
        embedding = await embed_single(task_description[:2000])
        embedding_str = format_embedding_for_pg(embedding)

        exemplars = await db.search_exemplars(
            embedding_str, project,
            task_type=task_type,
            limit=effective_limit,
            min_score=effective_min_score,
        )
    except Exception:
        logger.debug("Exemplar retrieval failed", exc_info=True)
        return "", []

    if not exemplars:
        return "", []

    # Increment usage counters
    ids = []
    for ex in exemplars:
        ids.append(ex.id)
        try:
            await db.increment_exemplar_usage(ex.id)
        except Exception:
            pass

    # Format as injectable context
    parts = []
    for ex in exemplars:
        score_str = f"{ex.outcome_score:.1f}" if ex.outcome_score else "?"
        parts.append(
            f"## Similar task (solved by {ex.model_source}, confidence: {score_str})\n"
            f"**Task**: {ex.task_description[:500]}\n"
            f"**Approach**: {ex.solution_approach[:1500]}"
        )

    prompt = (
        "<past_solutions>\n"
        "Here are similar problems solved previously — use these as reference:\n\n"
        + "\n\n---\n\n".join(parts)
        + "\n</past_solutions>"
    )
    return prompt, ids


async def update_outcome(
    db: Database,
    exemplar_id: int,
    success: bool,
) -> None:
    """Update outcome score based on observed results.

    Uses exponential moving average: new = 0.7 * old + 0.3 * signal.
    """
    try:
        await db.update_exemplar_outcome(exemplar_id, success)
    except Exception:
        logger.debug("Failed to update exemplar outcome", exc_info=True)


async def update_active_exemplars(
    db: Database,
    exemplar_ids: list[int],
    success: bool,
) -> None:
    """Update outcome for all exemplars used in the current turn."""
    if not exemplar_ids:
        return
    for eid in exemplar_ids:
        await update_outcome(db, eid, success)
