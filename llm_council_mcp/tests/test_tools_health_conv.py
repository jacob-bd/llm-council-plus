"""Tests for health and conversation management MCP tools."""

import json
import pytest
import respx
import httpx
from llm_council_mcp.server import create_server


@pytest.fixture
def server():
    return create_server(base_url="http://test:8001")


from llm_council_mcp.tests.conftest import get_json, get_text


# ── check_health ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_check_health_backend_reachable(server):
    """check_health returns 'reachable' when backend responds 200."""
    settings = {
        "council_models": ["openai:gpt-4.1", "anthropic:claude-sonnet-4"],
        "chairman_model": "anthropic:claude-opus-4",
        "execution_mode": "full",
        "search_provider": "duckduckgo",
        "openrouter_api_key_set": True,
        "anthropic_api_key_set": True,
        "ollama_base_url": "http://localhost:11434",
    }
    with respx.mock:
        respx.get("http://test:8001/api/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:8001/api/settings").mock(
            return_value=httpx.Response(200, json=settings)
        )
        result = await server.call_tool("providers", {"action": "health"})
        data = get_json(result)

    assert data["backend"] == "reachable"
    assert data["base_url"] == "http://test:8001"
    assert "openrouter" in data["configured_providers"]
    assert "anthropic" in data["configured_providers"]
    assert data["council_models"] == ["openai:gpt-4.1", "anthropic:claude-sonnet-4"]
    assert data["chairman_model"] == "anthropic:claude-opus-4"
    assert data["search_provider"] == "duckduckgo"
    assert data["ollama_url"] == "http://localhost:11434"


@pytest.mark.asyncio
async def test_check_health_backend_unreachable(server):
    """check_health returns 'unreachable' when backend cannot be reached."""
    with respx.mock:
        respx.get("http://test:8001/api/health").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = await server.call_tool("providers", {"action": "health"})
        data = get_json(result)

    assert data["backend"] == "unreachable"
    assert "error" in data
    assert data["base_url"] == "http://test:8001"


@pytest.mark.asyncio
async def test_check_health_no_configured_providers(server):
    """check_health returns empty configured_providers when no API keys are set."""
    settings = {
        "council_models": [],
        "chairman_model": None,
        "execution_mode": "full",
        "search_provider": "duckduckgo",
    }
    with respx.mock:
        respx.get("http://test:8001/api/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:8001/api/settings").mock(
            return_value=httpx.Response(200, json=settings)
        )
        result = await server.call_tool("providers", {"action": "health"})
        data = get_json(result)

    assert data["backend"] == "reachable"
    assert data["configured_providers"] == []


@pytest.mark.asyncio
async def test_check_health_settings_error(server):
    """check_health reports settings_error when settings endpoint fails."""
    with respx.mock:
        respx.get("http://test:8001/api/health").mock(
            return_value=httpx.Response(200, json={"status": "ok"})
        )
        respx.get("http://test:8001/api/settings").mock(
            return_value=httpx.Response(500, json={"detail": "Internal Server Error"})
        )
        result = await server.call_tool("providers", {"action": "health"})
        data = get_json(result)

    assert data["backend"] == "reachable"
    assert "settings_error" in data


# ── test_provider ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_test_provider_success(server):
    """test_provider returns success result from backend."""
    with respx.mock:
        respx.post("http://test:8001/api/settings/test-provider").mock(
            return_value=httpx.Response(200, json={
                "success": True,
                "message": "OpenAI connection successful. Model: gpt-4.1",
            })
        )
        result = await server.call_tool("providers", {"action": "test", "provider": "openai"})
        data = get_json(result)

    assert data["success"] is True
    assert "successful" in data["message"]


@pytest.mark.asyncio
async def test_test_provider_with_api_key(server):
    """test_provider passes api_key to backend when provided."""
    captured_body = {}

    def capture_request(request):
        captured_body.update(json.loads(request.content))
        return httpx.Response(200, json={"success": True, "message": "OK"})

    with respx.mock:
        respx.post("http://test:8001/api/settings/test-provider").mock(
            side_effect=capture_request
        )
        result = await server.call_tool("providers", {"action": "test", 
            "provider": "anthropic",
            "api_key": "sk-ant-test-key",
        })
        data = get_json(result)

    assert data["success"] is True
    assert captured_body["provider"] == "anthropic"
    assert captured_body["api_key"] == "sk-ant-test-key"


@pytest.mark.asyncio
async def test_test_provider_failure(server):
    """test_provider returns failure result when provider returns error."""
    with respx.mock:
        respx.post("http://test:8001/api/settings/test-provider").mock(
            return_value=httpx.Response(200, json={
                "success": False,
                "message": "Invalid API key",
            })
        )
        result = await server.call_tool("providers", {"action": "test", "provider": "groq"})
        data = get_json(result)

    assert data["success"] is False
    assert "Invalid API key" in data["message"]


@pytest.mark.asyncio
async def test_test_provider_connection_error(server):
    """test_provider returns success=False JSON when backend is unreachable."""
    with respx.mock:
        respx.post("http://test:8001/api/settings/test-provider").mock(
            side_effect=httpx.ConnectError("connection refused")
        )
        result = await server.call_tool("providers", {"action": "test", "provider": "openai"})
        data = get_json(result)

    assert data["success"] is False
    assert "message" in data


# ── list_conversations ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_conversations_empty(server):
    """list_conversations returns 'No conversations found' for empty list."""
    with respx.mock:
        respx.get("http://test:8001/api/conversations").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await server.call_tool("conversations", {"action": "list"})
        text = get_text(result)

    assert "No conversations found" in text


@pytest.mark.asyncio
async def test_list_conversations(server):
    """list_conversations shows count and details for each conversation."""
    conversations = [
        {
            "id": "conv-abc123",
            "title": "What is consciousness?",
            "message_count": 4,
            "created_at": "2026-05-10T14:30:00Z",
        },
        {
            "id": "conv-def456",
            "title": "Best programming languages",
            "message_count": 2,
            "created_at": "2026-05-09T10:00:00Z",
        },
    ]
    with respx.mock:
        respx.get("http://test:8001/api/conversations").mock(
            return_value=httpx.Response(200, json=conversations)
        )
        result = await server.call_tool("conversations", {"action": "list"})
        text = get_text(result)

    assert "Found 2 conversation(s)" in text
    assert "conv-abc123" in text
    assert "What is consciousness?" in text
    assert "conv-def456" in text
    assert "Best programming languages" in text
    assert "2026-05-10" in text


@pytest.mark.asyncio
async def test_list_conversations_untitled(server):
    """list_conversations shows '(untitled)' for conversations with no title."""
    conversations = [
        {
            "id": "conv-notitle",
            "title": "",
            "message_count": 1,
            "created_at": "2026-05-10T08:00:00Z",
        }
    ]
    with respx.mock:
        respx.get("http://test:8001/api/conversations").mock(
            return_value=httpx.Response(200, json=conversations)
        )
        result = await server.call_tool("conversations", {"action": "list"})
        text = get_text(result)

    assert "(untitled)" in text
    assert "conv-notitle" in text


# ── get_conversation ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_conversation(server):
    """get_conversation returns summarised JSON with title and messages."""
    conv = {
        "id": "conv-abc123",
        "title": "What is consciousness?",
        "created_at": "2026-05-10T14:30:00Z",
        "messages": [
            {
                "role": "user",
                "content": "What is consciousness? Please explain in detail.",
            },
            {
                "role": "assistant",
                "stage1": [
                    {"model": "openai:gpt-4.1", "response": "Consciousness is..."},
                    {"model": "anthropic:claude-sonnet-4", "response": "It involves..."},
                ],
                "stage3": {
                    "model": "anthropic:claude-opus-4",
                    "response": "The chairman synthesis: consciousness is a complex emergent property.",
                },
                "metadata": {"execution_mode": "full"},
            },
        ],
    }
    with respx.mock:
        respx.get("http://test:8001/api/conversations/conv-abc123").mock(
            return_value=httpx.Response(200, json=conv)
        )
        result = await server.call_tool("conversations", {"action": "get", "conversation_id": "conv-abc123"})
        data = get_json(result)

    assert data["id"] == "conv-abc123"
    assert data["title"] == "What is consciousness?"
    assert data["message_count"] == 2
    assert len(data["messages"]) == 2

    user_msg = data["messages"][0]
    assert user_msg["role"] == "user"
    assert "consciousness" in user_msg["content"]

    assistant_msg = data["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["stage1_model_count"] == 2
    assert assistant_msg["execution_mode"] == "full"
    assert "synthesis" in assistant_msg["chairman_synthesis"]


@pytest.mark.asyncio
async def test_get_conversation_not_found(server):
    """get_conversation returns error JSON for unknown conversation ID."""
    with respx.mock:
        respx.get("http://test:8001/api/conversations/bad-id").mock(
            return_value=httpx.Response(404, json={"detail": "Conversation not found"})
        )
        result = await server.call_tool("conversations", {"action": "get", "conversation_id": "bad-id"})
        data = get_json(result)

    assert data["status"] == "error"
    assert "not found" in data["message"].lower()


@pytest.mark.asyncio
async def test_get_conversation_truncates_long_user_content(server):
    """get_conversation truncates user messages at 200 chars."""
    long_content = "A" * 500
    conv = {
        "id": "conv-long",
        "title": "Long message test",
        "created_at": "2026-05-10T00:00:00Z",
        "messages": [
            {"role": "user", "content": long_content},
        ],
    }
    with respx.mock:
        respx.get("http://test:8001/api/conversations/conv-long").mock(
            return_value=httpx.Response(200, json=conv)
        )
        result = await server.call_tool("conversations", {"action": "get", "conversation_id": "conv-long"})
        data = get_json(result)

    user_msg = data["messages"][0]
    assert len(user_msg["content"]) == 200


@pytest.mark.asyncio
async def test_get_conversation_no_stage3(server):
    """get_conversation handles assistant messages without stage3 (chat_only mode)."""
    conv = {
        "id": "conv-chat-only",
        "title": "Chat only test",
        "created_at": "2026-05-10T00:00:00Z",
        "messages": [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "stage1": [{"model": "openai:gpt-4.1", "response": "Hi there!"}],
                "stage3": None,
                "metadata": {"execution_mode": "chat_only"},
            },
        ],
    }
    with respx.mock:
        respx.get("http://test:8001/api/conversations/conv-chat-only").mock(
            return_value=httpx.Response(200, json=conv)
        )
        result = await server.call_tool("conversations", {"action": "get", "conversation_id": "conv-chat-only"})
        data = get_json(result)

    assistant_msg = data["messages"][1]
    assert assistant_msg["role"] == "assistant"
    assert assistant_msg["stage1_model_count"] == 1
    assert assistant_msg["execution_mode"] == "chat_only"
    assert "chairman_synthesis" not in assistant_msg
