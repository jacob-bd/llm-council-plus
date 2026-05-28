"""Full settings backup MCP tool."""

from __future__ import annotations

import json
from typing import Any

from ..client import CouncilClient


def register(server, base_url: str) -> None:
    """Register config_backup tool."""

    @server.tool(description=(
        "Backup or restore full settings. action: 'export' (JSON backup), "
        "'import' (requires config_json), 'reset' (factory defaults — irreversible)."
    ))
    async def config_backup(
        action: str,
        config_json: str | dict[str, Any] | None = None,
    ) -> str:
        action = action.strip().lower()
        if action not in ("export", "import", "reset"):
            return "Error: action must be export, import, or reset."

        try:
            async with CouncilClient(base_url) as client:
                if action == "export":
                    data = await client.export_settings()
                    return json.dumps(data, indent=2)

                if action == "import":
                    if config_json is None:
                        return "Error: config_json is required for import."
                    if isinstance(config_json, dict):
                        data = config_json
                    else:
                        try:
                            data = json.loads(config_json)
                        except json.JSONDecodeError as e:
                            return f"Error: invalid JSON — {e}"
                    await client.import_settings(data)
                    return "Configuration imported successfully."

                await client.reset_settings()
                return "Configuration reset to defaults."
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)
