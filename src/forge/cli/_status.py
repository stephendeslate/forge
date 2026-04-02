"""Status command — show model/DB/config status."""

from __future__ import annotations

import asyncio

from rich.panel import Panel

from forge.config import settings
from forge.storage.database import Database

from ._helpers import console


def status_command() -> None:
    """Show model status and configuration."""
    import httpx

    console.print(Panel("[bold]Forge Status[/bold]", border_style="blue"))

    # Check Ollama
    try:
        resp = httpx.get(f"{settings.ollama.base_url}/api/tags", timeout=5)
        models = resp.json().get("models", [])
        model_names = [m["name"] for m in models]

        console.print(f"\n[bold]Ollama[/bold] ({settings.ollama.base_url})")
        heavy = settings.ollama.heavy_model
        fast = settings.ollama.fast_model
        embed = settings.ollama.embed_model

        for name, label in [(heavy, "heavy"), (fast, "fast"), (embed, "embed")]:
            available = any(name in m for m in model_names)
            icon = "[green]●[/green]" if available else "[red]○[/red]"
            console.print(f"  {icon} {name} ({label})")

        try:
            ps_resp = httpx.get(f"{settings.ollama.base_url}/api/ps", timeout=5)
            running = ps_resp.json().get("models", [])
            if running:
                console.print(f"\n  [bold]Running:[/bold]")
                for m in running:
                    size_gb = m.get("size_vram", 0) / (1024**3)
                    console.print(f"    {m['name']} ({size_gb:.1f} GiB VRAM)")
        except Exception:
            pass

    except httpx.ConnectError:
        console.print("[red]Ollama unreachable[/red] — is it running?")
    except Exception as e:
        console.print(f"[red]Ollama error:[/red] {e}")

    # NPU status
    console.print(f"\n[bold]NPU[/bold]")
    if settings.npu.enabled:
        try:
            npu_resp = httpx.get(f"{settings.npu.base_url.rstrip('/')}/models", timeout=5)
            npu_models = npu_resp.json().get("data", [])
            model_names = [m.get("id", "unknown") for m in npu_models]
            console.print(f"  [green]●[/green] Connected — {', '.join(model_names) or 'no models'}")
        except Exception:
            console.print(f"  [yellow]●[/yellow] Enabled but unreachable ({settings.npu.base_url})")
    else:
        console.print("  [dim]Disabled (set FORGE_NPU_ENABLED=true)[/dim]")

    # Database / RAG status
    console.print(f"\n[bold]Database[/bold]")
    try:
        db = Database()
        asyncio.run(_check_db_status(db))
    except Exception as e:
        console.print(f"  [red]Unavailable:[/red] {e}")

    console.print(f"\n[bold]Config[/bold]")
    console.print(f"  Default route: {settings.default_route}")
    console.print(f"  Streaming: {settings.streaming}")
    console.print(f"  Max history: {settings.max_history} turns")


async def _check_db_status(db: Database) -> None:
    try:
        await db.connect()
        count = await db.get_session_count()
        console.print(f"  [green]●[/green] Connected — {count} sessions")
        await db.close()
    except Exception:
        console.print("  [red]○[/red] Cannot connect — is PostgreSQL running?")
