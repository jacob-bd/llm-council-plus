# MCP Server Instructions (for agents)

This file documents the **agent-facing instructions** embedded in the MCP server (`llm_council_mcp/server.py`). If you change routing behavior, update **both** this file and `server.py`, plus the MCP-first section in [`skills/llm-council-api/SKILL.md`](../../skills/llm-council-api/SKILL.md).

---

## Primary rule

**When LLM Council Plus MCP tools are available in your session, use them.** Do not shell out to `curl` against `/api/*` for operations that have an MCP equivalent.

The REST API skill exists as a **fallback reference**, not the default path for interactive agents.

---

## Tool catalog (10 tools)

| Tool | Actions | Replaces (legacy) |
|------|---------|-------------------|
| `council_deliberate` | `stage1`, `stage2`, `stage3`, `full` | `run_stage1`, `run_stage2`, `run_stage3`, `run_deliberation` |
| `model_chat` | `quick`, `multi_turn` | `quick_chat`, `chat` |
| `advisor_debate` | _(direct params)_ | `run_advisor_debate` |
| `run_iterative_debate` | _(direct params: query, debate_rounds, critique_mode, etc.)_ | _(new in v0.7.0)_ |
| `council_settings` | `get`, `update`, `list_presets`, `save_preset`, `delete_preset`, `set_default_preset` | `get_council_config`, `configure_council` |
| `advisor_settings` | same preset actions + `get`, `update` | `get_advisor_config`, `configure_advisors` |
| `personas` | `list`, `get`, `update`, `reset` | `list_personas`, `get_persona`, `update_persona`, `reset_persona` |
| `conversations` | `list`, `get`, `progress` | `list_conversations`, `get_conversation` |
| `providers` | `list_models`, `health`, `test`, `set_api_key`, `set_search` | `list_models`, `check_health`, `test_provider`, `set_api_key`, `set_search_provider` |
| `config_backup` | `export`, `import`, `reset` | `export_config`, `import_config`, `reset_config` |

Server names vary by host: `llm-council-plus`, `llm-council`, `user-llm-council-plus`.

---

## Action selection guide

| You want to… | Tool + action |
|--------------|---------------|
| Check backend / configured providers | `providers` → `health` |
| Test an API key | `providers` → `test` |
| List available models | `providers` → `list_models` |
| Read/update council config | `council_settings` → `get` / `update` |
| Save/load council presets | `council_settings` → `save_preset`, `list_presets`, etc. |
| Full deliberation | `council_deliberate` → `full` |
| Stage 1 / 2 / 3 only | `council_deliberate` → `stage1` / `stage2` / `stage3` |
| One-shot model chat | `model_chat` → `quick` |
| Multi-turn single-model chat | `model_chat` → `multi_turn` |
| Run advisor debate | `advisor_debate` |
| Run multi-round council debate | `run_iterative_debate` |
| Check active run progress | `conversations` → `progress` |
| Manage personas | `personas` → `list` / `get` / `update` / `reset` |
| Read advisor defaults / presets | `advisor_settings` → `get` / `list_presets` |
| List/read conversations | `conversations` → `list` / `get` |
| Set search provider or API key | `providers` → `set_search` / `set_api_key` |
| Backup / restore settings | `config_backup` → `export` / `import` / `reset` |

---

## Use REST instead of MCP when

- **MCP unavailable** — connection refused, stale SSE session, tool not in list
- **Cron / CI / scripts** — no MCP transport
- **Raw SSE streams** — MCP returns consolidated results, not per-event SSE
- **Admin export with bearer token** — `GET /api/settings/export` with `LLM_COUNCIL_ADMIN_TOKEN` (manual admin)

Preset CRUD is available via MCP (`council_settings` / `advisor_settings` preset actions). Use REST `PUT /api/settings` only when MCP is unavailable.

---

## References

- Tool parameters: [TOOLS.md](TOOLS.md)
- Worked examples: [EXAMPLES.md](EXAMPLES.md)
- REST fallback + full routing table: [skills/llm-council-api/SKILL.md](../../skills/llm-council-api/SKILL.md)
