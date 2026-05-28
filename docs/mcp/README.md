# LLM Council Plus — MCP Server

LLM Council Plus exposes a Model Context Protocol (MCP) server that lets AI tools like Claude Code and Gemini CLI send questions directly to your council and retrieve deliberation results — all without opening a browser. Ask your AI assistant to run a full 3-stage deliberation, configure council members, initiate multi-round advisor debates, customize persona thinking styles, or fetch past conversations, and it talks to the backend on your behalf.

## Quick Start

### Option A: Local stdio (Standard for local development)
1. Install from project root: `pip install -e .`
2. Register: `claude mcp add llm-council python -m llm_council_mcp`

### Option B: Remote SSE (Zero-install for containers / servers)
1. Run backend container (exposing port `8001`).
2. Register: `claude mcp add llm-council --url http://yourserver.com:8001/mcp/sse`

---

## Choose Your Setup

| Scenario | Guide |
|----------|-------|
| Running Council locally on your machine | [SETUP-LOCAL.md](SETUP-LOCAL.md) |
| Council is on a remote server and you have Python locally | [SETUP-LOCAL.md](SETUP-LOCAL.md) (remote backend section) |
| Council is on a remote server and you want zero local install | [SETUP-REMOTE.md](SETUP-REMOTE.md) |
| Deciding between stdio and SSE transports | [CHOOSING-TRANSPORT.md](CHOOSING-TRANSPORT.md) |

## Tools Reference

See [TOOLS.md](TOOLS.md) for all **9 tools** with actions, parameters, and examples.

`GET /api/health` includes `"mcp": {"tools": 9, "sse_url": "..."}` so clients can verify the expected tool count.

**Agents:** MCP tools take priority over REST/curl when both are available. See [INSTRUCTIONS.md](INSTRUCTIONS.md) and the MCP-first section in [`skills/llm-council-api/SKILL.md`](../../skills/llm-council-api/SKILL.md).

## Examples

See [EXAMPLES.md](EXAMPLES.md) for real-world usage walkthroughs.

When MCP tools or settings fields change, keep [TOOLS.md](TOOLS.md), [EXAMPLES.md](EXAMPLES.md), and [`skills/llm-council-api/SKILL.md`](../../skills/llm-council-api/SKILL.md) aligned — see [`docs/DOC-SYNC.md`](../DOC-SYNC.md).

## MCP Not Working? Use the REST API Skill Instead

If the MCP server is unavailable, the SSE session is stale, or you prefer direct HTTP access, install the **`llm-council-api` Claude Code skill**. It gives Claude the same capabilities (configure council, run deliberations, list models, check health) via the REST API — no MCP required.

```bash
mkdir -p ~/.claude/skills/llm-council-api
curl -o ~/.claude/skills/llm-council-api/SKILL.md \
  https://raw.githubusercontent.com/jacob-bd/llm-council-plus/main/skills/llm-council-api/SKILL.md
```

See [`skills/llm-council-api/SKILL.md`](../../skills/llm-council-api/SKILL.md) for the full reference including examples, error handling, and troubleshooting.
