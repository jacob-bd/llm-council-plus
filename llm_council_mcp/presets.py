"""Preset list/save/delete helpers for MCP settings tools."""

from __future__ import annotations

import uuid
from typing import Any

from .client import CouncilClient

MAX_COUNCIL_PRESETS = 20
MAX_ADVISOR_PRESETS = 20


def _new_preset_id() -> str:
    return str(uuid.uuid4())


def _clear_default(presets: list[dict[str, Any]]) -> None:
    for preset in presets:
        preset["is_default"] = False


async def list_council_presets(client: CouncilClient) -> list[dict[str, Any]]:
    settings = await client.get_settings()
    return list(settings.get("council_presets") or [])


async def list_advisor_presets(client: CouncilClient) -> list[dict[str, Any]]:
    settings = await client.get_settings()
    return list(settings.get("advisor_presets") or [])


async def save_council_preset(
    client: CouncilClient,
    *,
    name: str,
    council_models: list[str],
    chairman_model: str = "",
    preset_id: str | None = None,
    is_default: bool = False,
) -> dict[str, Any]:
    if not name.strip():
        raise ValueError("Preset name is required.")
    if not council_models:
        raise ValueError("council_models must include at least one model.")

    settings = await client.get_settings()
    presets = list(settings.get("council_presets") or [])
    preset_id = (preset_id or "").strip() or _new_preset_id()

    entry = {
        "id": preset_id,
        "name": name.strip(),
        "council_models": council_models,
        "chairman_model": chairman_model or "",
        "is_default": is_default,
        "last_used_at": None,
    }

    idx = next((i for i, p in enumerate(presets) if p.get("id") == preset_id), None)
    if idx is None:
        if len(presets) >= MAX_COUNCIL_PRESETS:
            raise ValueError(f"Maximum of {MAX_COUNCIL_PRESETS} council presets reached.")
        presets.append(entry)
        target_idx = len(presets) - 1
    else:
        presets[idx] = {**presets[idx], **entry}
        target_idx = idx

    if is_default:
        _clear_default(presets)
        presets[target_idx]["is_default"] = True

    await client.update_settings(council_presets=presets)
    return presets[target_idx]


async def delete_council_preset(client: CouncilClient, preset_id: str) -> None:
    preset_id = preset_id.strip()
    settings = await client.get_settings()
    presets = list(settings.get("council_presets") or [])
    next_presets = [p for p in presets if p.get("id") != preset_id]
    if len(next_presets) == len(presets):
        raise ValueError(f"Council preset '{preset_id}' not found.")
    await client.update_settings(council_presets=next_presets)


async def set_default_council_preset(client: CouncilClient, preset_id: str) -> dict[str, Any]:
    preset_id = preset_id.strip()
    settings = await client.get_settings()
    presets = list(settings.get("council_presets") or [])
    match = next((p for p in presets if p.get("id") == preset_id), None)
    if match is None:
        raise ValueError(f"Council preset '{preset_id}' not found.")
    _clear_default(presets)
    match["is_default"] = True
    await client.update_settings(council_presets=presets)
    return match


async def save_advisor_preset(
    client: CouncilClient,
    *,
    name: str,
    persona_ids: list[str],
    mode: str = "simple",
    default_model: str = "",
    tiebreaker_model: str = "",
    model_assignments: dict[str, str] | None = None,
    max_rounds: int = 3,
    search_provider: str | None = None,
    preset_id: str | None = None,
    is_default: bool = False,
) -> dict[str, Any]:
    if not name.strip():
        raise ValueError("Preset name is required.")
    if len(persona_ids) < 2:
        raise ValueError("persona_ids must include at least 2 personas.")

    settings = await client.get_settings()
    presets = list(settings.get("advisor_presets") or [])
    preset_id = (preset_id or "").strip() or _new_preset_id()

    entry = {
        "id": preset_id,
        "name": name.strip(),
        "persona_ids": persona_ids,
        "mode": mode if mode in ("simple", "advanced") else "simple",
        "default_model": default_model or "",
        "tiebreaker_model": tiebreaker_model or "",
        "model_assignments": model_assignments,
        "max_rounds": max_rounds,
        "search_provider": search_provider,
        "is_default": is_default,
        "last_used_at": None,
    }

    idx = next((i for i, p in enumerate(presets) if p.get("id") == preset_id), None)
    if idx is None:
        if len(presets) >= MAX_ADVISOR_PRESETS:
            raise ValueError(f"Maximum of {MAX_ADVISOR_PRESETS} advisor presets reached.")
        presets.append(entry)
        target_idx = len(presets) - 1
    else:
        presets[idx] = {**presets[idx], **entry}
        target_idx = idx

    if is_default:
        _clear_default(presets)
        presets[target_idx]["is_default"] = True

    await client.update_settings(advisor_presets=presets)
    return presets[target_idx]


async def delete_advisor_preset(client: CouncilClient, preset_id: str) -> None:
    preset_id = preset_id.strip()
    settings = await client.get_settings()
    presets = list(settings.get("advisor_presets") or [])
    next_presets = [p for p in presets if p.get("id") != preset_id]
    if len(next_presets) == len(presets):
        raise ValueError(f"Advisor preset '{preset_id}' not found.")
    await client.update_settings(advisor_presets=next_presets)


async def set_default_advisor_preset(client: CouncilClient, preset_id: str) -> dict[str, Any]:
    preset_id = preset_id.strip()
    settings = await client.get_settings()
    presets = list(settings.get("advisor_presets") or [])
    match = next((p for p in presets if p.get("id") == preset_id), None)
    if match is None:
        raise ValueError(f"Advisor preset '{preset_id}' not found.")
    _clear_default(presets)
    match["is_default"] = True
    await client.update_settings(advisor_presets=presets)
    return match
