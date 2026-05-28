# stdio Setup (Local or Remote Backend)

This guide covers installing the MCP server as a local process (stdio transport). It works whether your Council backend is on your laptop or on a remote server.

If you want zero local installation, you can connect directly to the remote server's built-in SSE endpoint on port 8001 at `/mcp/sse` (see [SETUP-REMOTE.md](SETUP-REMOTE.md) instead). If you are unsure which to choose, see [CHOOSING-TRANSPORT.md](CHOOSING-TRANSPORT.md).

---

## Prerequisites

- **Python 3.10+** — check with `python --version` or `python3 --version`
- **LLM Council Plus backend running** — local at `http://localhost:8001`, or accessible at a remote URL

---

## Step 1: Install the MCP package

Run this from the project root (the directory containing `pyproject.toml`):

```bash
# Using pip
pip install -e .

# Or using uv (recommended if you already use uv for the backend)
uv tool install .
```

This installs `llm_council_mcp` as both a Python package and a runnable module (`python -m llm_council_mcp`).

After install, verify the module is importable:

```bash
python -m llm_council_mcp --help
```

---

## Step 2: Register with Claude Code

**Local backend (default):**
```bash
claude mcp add llm-council python -m llm_council_mcp
```

**Remote backend:**
```bash
claude mcp add llm-council python -m llm_council_mcp --base-url https://yourserver.com:8001
```

Verify it was registered:
```bash
claude mcp list
```

You should see `llm-council` in the output.

---

## Step 3: Register with Gemini CLI

**Local backend:**
```bash
gemini mcp add llm-council --command "python -m llm_council_mcp"
```

**Remote backend:**
```bash
gemini mcp add llm-council --command "python -m llm_council_mcp --base-url https://yourserver.com:8001"
```

---

## Step 4: Verify it works

Start a new conversation in Claude Code or Gemini CLI and ask:

> "Check the council health"

The AI will call `providers` with action `health`. A successful response looks like:

```json
{
  "backend": "reachable",
  "base_url": "http://localhost:8001",
  "council_models": ["openai:gpt-4.1", "..."],
  "configured_providers": ["openai", "anthropic"]
}
```

Confirm the backend advertises 9 MCP tools: `GET /api/health` → `"mcp": {"tools": 9}`.

---

## Remote backend (detailed)

When the backend is on a remote server, the MCP server runs locally on your machine but all API calls go to the remote host over HTTPS. Your local machine does not need to run the Council backend — only the MCP shim.

```bash
# Install (same as above, from project root)
pip install -e .

# Register with remote URL
claude mcp add llm-council python -m llm_council_mcp \
  --base-url https://yourserver.com:8001
```

The `--base-url` flag tells the MCP server where to find the Council API. It replaces `http://localhost:8001` in all outbound requests.

---

## Troubleshooting

**"Backend not running" or "Connection refused"**
- Make sure the backend is started: `uv run python -m backend.main` (local) or verify the remote URL is accessible
- Confirm port 8001 is open: `curl http://localhost:8001/api/health` (local) or `curl https://yourserver.com:8001/api/health` (remote)

**"python: command not found" or "module not found"**
- Use `python3` instead of `python` if your system requires it
- Re-run `pip install -e .` from the project root to ensure the package is installed in the active Python environment
- If using uv: `uv tool install --force .`

**Claude Code says the tool is unavailable**
- Run `claude mcp list` to confirm `llm-council` is registered
- Try removing and re-adding: `claude mcp remove llm-council` then the add command above
- Restart Claude Code after registration changes

**"Connection refused" for remote backend**
- Confirm the backend URL is correct and port 8001 is open in the server's firewall
- Test from your machine: `curl https://yourserver.com:8001/api/health`
