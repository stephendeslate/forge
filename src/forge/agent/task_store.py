"""In-memory task tracking for agent self-organization."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum


class TaskStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    DELETED = "deleted"


@dataclass
class Task:
    id: str
    subject: str
    description: str
    status: TaskStatus = TaskStatus.PENDING
    active_form: str | None = None
    blocked_by: list[str] = field(default_factory=list)
    blocks: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class TaskStore:
    """In-memory task store with auto-incrementing IDs."""

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._counter: int = 0

    def create(
        self,
        subject: str,
        description: str,
        *,
        active_form: str | None = None,
    ) -> Task:
        """Create a new task with auto-incremented ID."""
        self._counter += 1
        task_id = f"t{self._counter}"
        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
        )
        self._tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs) -> Task | None:
        task = self._tasks.get(task_id)
        if task is None:
            return None

        if "status" in kwargs:
            task.status = kwargs["status"]
        if "subject" in kwargs:
            task.subject = kwargs["subject"]
        if "description" in kwargs:
            task.description = kwargs["description"]
        if "active_form" in kwargs:
            task.active_form = kwargs["active_form"]
        if "metadata" in kwargs:
            task.metadata.update(kwargs["metadata"])
        if "add_blocked_by" in kwargs:
            for dep_id in kwargs["add_blocked_by"]:
                if dep_id not in task.blocked_by:
                    task.blocked_by.append(dep_id)
                # Set reverse link
                dep = self._tasks.get(dep_id)
                if dep and task_id not in dep.blocks:
                    dep.blocks.append(task_id)
        if "add_blocks" in kwargs:
            for dep_id in kwargs["add_blocks"]:
                if dep_id not in task.blocks:
                    task.blocks.append(dep_id)
                dep = self._tasks.get(dep_id)
                if dep and task_id not in dep.blocked_by:
                    dep.blocked_by.append(task_id)

        return task

    def list_all(self) -> list[Task]:
        """List all non-deleted tasks."""
        return [t for t in self._tasks.values() if t.status != TaskStatus.DELETED]

    def list_open(self) -> list[Task]:
        """List pending and in-progress tasks."""
        return [
            t for t in self._tasks.values()
            if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS)
        ]

    def get_active(self) -> Task | None:
        """Get the first in-progress task."""
        for t in self._tasks.values():
            if t.status == TaskStatus.IN_PROGRESS:
                return t
        return None

    def to_prompt(self) -> str:
        """Render tasks as markdown for system prompt injection."""
        tasks = self.list_all()
        if not tasks:
            return ""

        lines = ["## Current tasks\n"]
        for t in tasks:
            icon = {
                TaskStatus.PENDING: "[ ]",
                TaskStatus.IN_PROGRESS: "[~]",
                TaskStatus.COMPLETED: "[x]",
            }.get(t.status, "[-]")
            blocked = f" (blocked by {','.join(t.blocked_by)})" if t.blocked_by else ""
            lines.append(f"- {icon} {t.id}: {t.subject}{blocked}")

        active = self.get_active()
        if active and active.active_form:
            lines.append(f"\nCurrently: {active.active_form}")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialize to JSON for persistence."""
        data = {
            "counter": self._counter,
            "tasks": [
                {
                    "id": t.id,
                    "subject": t.subject,
                    "description": t.description,
                    "status": t.status.value,
                    "active_form": t.active_form,
                    "blocked_by": t.blocked_by,
                    "blocks": t.blocks,
                    "metadata": t.metadata,
                }
                for t in self._tasks.values()
            ],
        }
        return json.dumps(data, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> TaskStore:
        """Deserialize from JSON."""
        data = json.loads(raw)
        store = cls()
        store._counter = data.get("counter", 0)
        for t in data.get("tasks", []):
            task = Task(
                id=t["id"],
                subject=t["subject"],
                description=t["description"],
                status=TaskStatus(t["status"]),
                active_form=t.get("active_form"),
                blocked_by=t.get("blocked_by", []),
                blocks=t.get("blocks", []),
                metadata=t.get("metadata", {}),
            )
            store._tasks[task.id] = task
        return store
