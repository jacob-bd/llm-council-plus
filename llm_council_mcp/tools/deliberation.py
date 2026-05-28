"""Council deliberation and direct model chat MCP tools."""

from __future__ import annotations

import json

from ..client import CouncilClient
from ..stream_buffer import buffer_stage1, buffer_stage2, buffer_stage3


def register(server, base_url: str) -> None:
    """Register council_deliberate and model_chat tools."""

    @server.tool(description=(
        "Run council deliberation. action: 'stage1' (individual responses), "
        "'stage2' (peer rankings), 'stage3' (chairman synthesis only), "
        "'full' (all 3 stages). Pass conversation_id to continue a thread. "
        "web_search enriches the query. models overrides council members for 'full' only."
    ))
    async def council_deliberate(
        action: str,
        query: str,
        web_search: bool = False,
        conversation_id: str | None = None,
        models: list[str] | None = None,
    ) -> str:
        action = action.strip().lower()
        if action not in ("stage1", "stage2", "stage3", "full"):
            return "Error: action must be stage1, stage2, stage3, or full."

        try:
            async with CouncilClient(base_url) as client:
                if not conversation_id:
                    conv = await client.create_conversation()
                    conversation_id = conv["id"]

                if action == "stage1":
                    events = client.stream_message(
                        conversation_id, query, web_search=web_search, execution_mode="chat_only"
                    )
                    result, _ = await buffer_stage1(events, conversation_id, query)
                    return json.dumps(result, indent=2)

                if action == "stage2":
                    events = client.stream_message(
                        conversation_id, query, web_search=False, execution_mode="chat_ranking"
                    )
                    _, remaining = await buffer_stage1(events, conversation_id, query)
                    result, _ = await buffer_stage2(remaining, conversation_id)
                    return json.dumps(result, indent=2)

                if action == "stage3":
                    events = client.stream_message(
                        conversation_id, query, web_search=False, execution_mode="full"
                    )
                    _, after1 = await buffer_stage1(events, conversation_id, query)
                    _, after2 = await buffer_stage2(after1, conversation_id)
                    result = await buffer_stage3(after2, conversation_id)
                    return json.dumps(result, indent=2)

                events = client.stream_message(
                    conversation_id, query,
                    web_search=web_search, execution_mode="full",
                    council_models=models,
                )
                stage1, after1 = await buffer_stage1(events, conversation_id, query)
                stage2, after2 = await buffer_stage2(after1, conversation_id)
                stage3 = await buffer_stage3(after2, conversation_id)

            return json.dumps({
                "conversation_id": conversation_id,
                "query": query,
                "stage1": stage1,
                "stage2": stage2,
                "stage3": stage3,
                "chairman_answer": stage3.get("synthesis"),
            }, indent=2)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    @server.tool(description=(
        "Chat with a single model. action: 'quick' (one-shot, no memory) or "
        "'multi_turn' (pass conversation_id from prior response to continue). "
        "model must include provider prefix, e.g. openai:gpt-4.1 or ollama:llama3."
    ))
    async def model_chat(
        action: str,
        query: str,
        model: str,
        conversation_id: str | None = None,
        web_search: bool = False,
    ) -> str:
        action = action.strip().lower()
        if action not in ("quick", "multi_turn"):
            return "Error: action must be quick or multi_turn."

        try:
            async with CouncilClient(base_url) as client:
                if action == "quick":
                    result = await client.ask(
                        content=query,
                        models=[model],
                        web_search=web_search,
                        execution_mode="chat_only",
                    )
                    return json.dumps({
                        "model": result.get("model", model),
                        "response": result.get("response"),
                        "error": result.get("error"),
                        "web_search_used": web_search,
                    }, indent=2)

                if not conversation_id:
                    conv = await client.create_conversation()
                    conversation_id = conv["id"]
                events = client.stream_message(
                    conversation_id, query,
                    web_search=web_search, execution_mode="chat_only",
                    council_models=[model],
                )
                result, _ = await buffer_stage1(events, conversation_id, query)

            responses = result.get("results", [])
            first = responses[0] if responses else {}
            return json.dumps({
                "conversation_id": conversation_id,
                "model": first.get("model", model),
                "response": first.get("response"),
                "error": first.get("error"),
                "web_search_used": web_search,
            }, indent=2)
        except Exception as exc:
            return json.dumps({"status": "error", "message": str(exc)}, indent=2)
