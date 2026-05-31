"""Conversation MCP tool."""

from __future__ import annotations

import json

from ..client import CouncilClient


def register(server, base_url: str) -> None:
    """Register conversations tool."""

    @server.tool(description=(
        "Manage saved conversations. action: 'list' (titles and IDs), "
        "'get' (full summary by conversation_id), or "
        "'progress' (live progress of an active streaming run by conversation_id)."
    ))
    async def conversations(
        action: str,
        conversation_id: str | None = None,
    ) -> str:
        action = action.strip().lower()
        if action not in ("list", "get", "progress"):
            return "Error: action must be list, get, or progress."

        try:
            async with CouncilClient(base_url) as client:
                if action == "list":
                    items = await client.list_conversations()
                    if not items:
                        return "No conversations found."
                    lines = [f"Found {len(items)} conversation(s):"]
                    for conv in items:
                        title = conv.get("title") or "(untitled)"
                        conv_id = conv.get("id", "unknown")
                        count = conv.get("message_count", "?")
                        created = conv.get("created_at", "")[:10]
                        lines.append(f"  • [{conv_id}] {title} — {count} message(s), created {created}")
                    return "\n".join(lines)

                if action == "progress":
                    if not conversation_id:
                        return "Error: conversation_id is required for progress."
                    progress = await client.get_conversation_progress(conversation_id)
                    return json.dumps(progress, indent=2)

                if not conversation_id:
                    return "Error: conversation_id is required for get."

                conv = await client.get_conversation(conversation_id)
                messages = conv.get("messages", [])
                summary = {
                    "id": conv.get("id"),
                    "title": conv.get("title"),
                    "created_at": conv.get("created_at"),
                    "message_count": len(messages),
                    "messages": [],
                }
                for msg in messages:
                    role = msg.get("role")
                    if role == "user":
                        summary["messages"].append({
                            "role": "user",
                            "content": msg.get("content", "")[:200],
                        })
                    elif role == "assistant":
                        stage3 = msg.get("stage3")
                        stage1 = msg.get("stage1", [])
                        mode = msg.get("metadata", {}).get("execution_mode", "unknown")
                        entry = {
                            "role": "assistant",
                            "execution_mode": mode,
                            "stage1_model_count": len(stage1),
                        }
                        if stage3:
                            entry["chairman_synthesis"] = (stage3.get("response") or "")[:500]
                        summary["messages"].append(entry)
                return json.dumps(summary, indent=2)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)
