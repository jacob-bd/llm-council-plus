"""Council settings and preset MCP tool."""

from __future__ import annotations

import json
from typing import Any

from ..client import CouncilClient
from .. import presets as preset_ops


def register(server, base_url: str) -> None:
    """Register council_settings tool."""

    @server.tool(description=(
        "Manage council configuration. action: 'get' (current config + presets), "
        "'update' (members/chairman/temps/mode/prompts/provider toggles), "
        "'list_presets', 'save_preset', 'delete_preset', 'set_default_preset'. "
        "For save_preset provide name + council_models; optional chairman_model, "
        "preset_id (omit to create), is_default."
    ))
    async def council_settings(
        action: str,
        models: list[str] | None = None,
        chairman: str | None = None,
        council_temperature: float | None = None,
        chairman_temperature: float | None = None,
        stage2_temperature: float | None = None,
        execution_mode: str | None = None,
        stage1_prompt: str | None = None,
        stage2_prompt: str | None = None,
        stage3_prompt: str | None = None,
        enabled_providers: dict[str, bool] | None = None,
        direct_provider_toggles: dict[str, bool] | None = None,
        preset_id: str | None = None,
        preset_name: str | None = None,
        council_models: list[str] | None = None,
        chairman_model: str | None = None,
        is_default: bool = False,
        critique_mode: str | None = None,
        debate_rounds: int | None = None,
        auto_converge: bool | None = None,
        convergence_threshold: int | None = None,
    ) -> str:
        action = action.strip().lower()
        valid = ("get", "update", "list_presets", "save_preset", "delete_preset", "set_default_preset")
        if action not in valid:
            return f"Error: action must be one of: {', '.join(valid)}."

        try:
            async with CouncilClient(base_url) as client:
                if action == "get":
                    settings = await client.get_settings()
                    config = {
                        "council_models": settings.get("council_models", []),
                        "chairman_model": settings.get("chairman_model"),
                        "council_temperature": settings.get("council_temperature"),
                        "chairman_temperature": settings.get("chairman_temperature"),
                        "stage2_temperature": settings.get("stage2_temperature"),
                        "execution_mode": settings.get("execution_mode"),
                        "search_provider": settings.get("search_provider"),
                        "critique_mode": settings.get("critique_mode"),
                        "debate_rounds": settings.get("debate_rounds"),
                        "auto_converge": settings.get("auto_converge"),
                        "convergence_threshold": settings.get("convergence_threshold"),
                        "council_presets": settings.get("council_presets", []),
                    }
                    return json.dumps(config, indent=2)

                if action == "list_presets":
                    items = await preset_ops.list_council_presets(client)
                    return json.dumps(items, indent=2)

                if action == "save_preset":
                    if not preset_name:
                        return "Error: preset_name is required for save_preset."
                    models_for_preset = council_models if council_models is not None else models
                    if not models_for_preset:
                        return "Error: council_models is required for save_preset."
                    saved = await preset_ops.save_council_preset(
                        client,
                        name=preset_name,
                        council_models=models_for_preset,
                        chairman_model=chairman_model or chairman or "",
                        preset_id=preset_id,
                        is_default=is_default,
                    )
                    return json.dumps({"status": "saved", "preset": saved}, indent=2)

                if action == "delete_preset":
                    if not preset_id:
                        return "Error: preset_id is required for delete_preset."
                    await preset_ops.delete_council_preset(client, preset_id)
                    return json.dumps({"status": "deleted", "preset_id": preset_id}, indent=2)

                if action == "set_default_preset":
                    if not preset_id:
                        return "Error: preset_id is required for set_default_preset."
                    preset = await preset_ops.set_default_council_preset(client, preset_id)
                    return json.dumps({"status": "default_set", "preset": preset}, indent=2)

                updates: dict[str, Any] = {}
                if models is not None:
                    if not (1 <= len(models) <= 8):
                        return f"Error: council requires 1-8 models, got {len(models)}."
                    updates["council_models"] = models
                if chairman is not None:
                    updates["chairman_model"] = chairman
                if council_temperature is not None:
                    updates["council_temperature"] = council_temperature
                if chairman_temperature is not None:
                    updates["chairman_temperature"] = chairman_temperature
                if stage2_temperature is not None:
                    updates["stage2_temperature"] = stage2_temperature
                if execution_mode is not None:
                    if execution_mode not in ("full", "chat_ranking", "chat_only"):
                        return "Error: execution_mode must be full, chat_ranking, or chat_only."
                    updates["execution_mode"] = execution_mode
                if stage1_prompt is not None:
                    updates["stage1_prompt"] = stage1_prompt
                if stage2_prompt is not None:
                    updates["stage2_prompt"] = stage2_prompt
                if stage3_prompt is not None:
                    updates["stage3_prompt"] = stage3_prompt
                if enabled_providers is not None:
                    updates["enabled_providers"] = enabled_providers
                if direct_provider_toggles is not None:
                    updates["direct_provider_toggles"] = direct_provider_toggles
                if critique_mode is not None:
                    critique_mode = critique_mode.strip().lower()
                    if critique_mode not in ("freeform", "paragraph", "claim"):
                        return "Error: critique_mode must be freeform, paragraph, or claim."
                    updates["critique_mode"] = critique_mode
                if debate_rounds is not None:
                    if not (1 <= debate_rounds <= 5):
                        return "Error: debate_rounds must be between 1 and 5."
                    updates["debate_rounds"] = debate_rounds
                if auto_converge is not None:
                    updates["auto_converge"] = auto_converge
                if convergence_threshold is not None:
                    if not (1 <= convergence_threshold <= 3):
                        return "Error: convergence_threshold must be 1, 2, or 3."
                    updates["convergence_threshold"] = convergence_threshold

                if not updates:
                    return "Error: no update fields provided."

                await client.update_settings(**updates)
                return json.dumps({"status": "updated", "fields": list(updates.keys())}, indent=2)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)
