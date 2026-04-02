"""Schema migration runner — auto-discovers and applies SQL migrations."""

from __future__ import annotations

import re
from pathlib import Path

import asyncpg

from forge.log import get_logger

logger = get_logger(__name__)

# SQL files live alongside this __init__.py
MIGRATIONS_DIR = Path(__file__).parent
ADVISORY_LOCK_ID = 42424242

_VERSION_RE = re.compile(r"^(\d{3})_(.+)\.sql$")


def list_available_migrations() -> list[tuple[int, str, Path]]:
    """Discover migration files from the migrations directory.

    Returns list of (version, name, path) sorted by version.
    """
    migrations: list[tuple[int, str, Path]] = []
    if not MIGRATIONS_DIR.is_dir():
        return migrations

    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        m = _VERSION_RE.match(f.name)
        if m:
            version = int(m.group(1))
            name = m.group(2)
            migrations.append((version, name, f))

    return sorted(migrations, key=lambda t: t[0])


def parse_migration_sql(path: Path) -> tuple[str, str]:
    """Parse a migration file into (forward_sql, rollback_sql).

    Everything before '-- rollback' is forward SQL.
    Everything after is rollback SQL. Rollback is optional.
    """
    content = path.read_text()
    parts = re.split(r"^-- rollback\s*$", content, maxsplit=1, flags=re.MULTILINE)
    forward = parts[0].strip()
    rollback = parts[1].strip() if len(parts) > 1 else ""
    return forward, rollback


async def _ensure_migration_table(conn: asyncpg.Connection) -> None:
    """Create the schema_migrations tracking table if it doesn't exist."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version     INTEGER PRIMARY KEY,
            name        TEXT NOT NULL,
            applied_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
    """)


async def get_current_version(pool: asyncpg.Pool) -> int:
    """Return the highest applied migration version, or 0 if none."""
    async with pool.acquire() as conn:
        await _ensure_migration_table(conn)
        row = await conn.fetchrow(
            "SELECT COALESCE(MAX(version), 0) AS v FROM schema_migrations"
        )
        return row["v"]


async def run_migrations(pool: asyncpg.Pool) -> list[str]:
    """Run all pending migrations. Returns list of applied migration names.

    Uses an advisory lock to prevent concurrent migration runs.
    Each migration runs in its own transaction.
    """
    available = list_available_migrations()
    if not available:
        return []

    applied: list[str] = []

    async with pool.acquire() as conn:
        # Advisory lock — prevents concurrent migration runs
        await conn.execute(f"SELECT pg_advisory_lock({ADVISORY_LOCK_ID})")
        try:
            await _ensure_migration_table(conn)

            current = await conn.fetchval(
                "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
            )

            for version, name, path in available:
                if version <= current:
                    continue

                forward_sql, _ = parse_migration_sql(path)
                logger.info("Applying migration %03d_%s", version, name)

                async with conn.transaction():
                    await conn.execute(forward_sql)
                    await conn.execute(
                        "INSERT INTO schema_migrations (version, name) VALUES ($1, $2)",
                        version,
                        name,
                    )

                applied.append(f"{version:03d}_{name}")
        finally:
            await conn.execute(f"SELECT pg_advisory_unlock({ADVISORY_LOCK_ID})")

    return applied


async def rollback_migration(pool: asyncpg.Pool, version: int) -> str:
    """Rollback a specific migration version. Returns the migration name.

    Raises ValueError if the version is not applied or has no rollback SQL.
    """
    available = {v: (n, p) for v, n, p in list_available_migrations()}
    if version not in available:
        raise ValueError(f"Migration {version} not found")

    name, path = available[version]
    _, rollback_sql = parse_migration_sql(path)
    if not rollback_sql:
        raise ValueError(f"Migration {version:03d}_{name} has no rollback section")

    async with pool.acquire() as conn:
        await conn.execute(f"SELECT pg_advisory_lock({ADVISORY_LOCK_ID})")
        try:
            row = await conn.fetchrow(
                "SELECT version FROM schema_migrations WHERE version = $1", version
            )
            if not row:
                raise ValueError(f"Migration {version} is not applied")

            async with conn.transaction():
                await conn.execute(rollback_sql)
                await conn.execute(
                    "DELETE FROM schema_migrations WHERE version = $1", version
                )
        finally:
            await conn.execute(f"SELECT pg_advisory_unlock({ADVISORY_LOCK_ID})")

    return f"{version:03d}_{name}"
