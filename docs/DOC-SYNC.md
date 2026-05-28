# Documentation Sync Checklist

When you change LLM Council Plus behavior, **update every surface in the same PR** (or same release commit). Do not ship code without syncing docs.

Canonical API reference: [`skills/llm-council-api/SKILL.md`](../skills/llm-council-api/SKILL.md)

---

## Always update (versioned releases)

When bumping version, update **all three together** (see `AGENTS.md` → Versioning Checklist):

| File | What to update |
|------|----------------|
| `CHANGELOG.md` | `## [x.y.z]` or `## [Unreleased]` entries |
| `frontend/src/components/Sidebar.jsx` | `sidebar-version` |
| `skills/llm-council-api/SKILL.md` | YAML frontmatter `version:` |

---

## Per change type — file matrix

### Settings / API fields (`backend/settings.py`, `PUT /api/settings`, `GET /api/settings`)

| File | Action |
|------|--------|
| `skills/llm-council-api/SKILL.md` | Field table, curl examples, GET response keys |
| `docs/mcp/TOOLS.md` | MCP tools (`council_settings`, `council_deliberate`, etc.) |
| `docs/mcp/EXAMPLES.md` | Example payloads if behavior changes |
| `AGENTS.md` | Settings UI section, storage notes |
| `README.md` | Configuration / first-time setup if user-facing |
| `docs/QUICKSTART.md` | Setup steps |
| `llm_council_mcp/tools/*.py` | Tool descriptions + returned JSON shape |
| `llm_council_mcp/tests/test_tools_*.py` | Assertions on new fields |
| `CHANGELOG.md` | Added/changed/fixed |

### Council behavior (members, chairman, streaming, execution modes)

| File | Action |
|------|--------|
| `skills/llm-council-api/SKILL.md` | Council endpoints, overrides, `council_models` / `chairman_model` |
| `docs/mcp/TOOLS.md` | `council_settings`, `council_deliberate`, health via `providers` |
| `AGENTS.md` | Architecture, execution modes, frontend components |
| `README.md` | Council overview if flow changes |
| `CHANGELOG.md` | |

### Advisor behavior (debates, personas, presets, model pickers)

| File | Action |
|------|--------|
| `skills/llm-council-api/SKILL.md` | Debate stream, advisor settings, `advisor_presets` |
| `docs/mcp/TOOLS.md` | `advisor_debate`, `advisor_settings`, `personas` |
| `docs/mcp/EXAMPLES.md` | Advisor walkthroughs |
| `docs/QUICKSTART.md` | Advisor setup path |
| `AGENTS.md` | Advisor modules, UI components |
| `llm_council_mcp/tools/advisors.py` | Tool descriptions + `advisor_settings` payload |
| `CHANGELOG.md` | |

### Provider / model routing (new prefix, NVIDIA, Ollama, etc.)

| File | Action |
|------|--------|
| `skills/llm-council-api/SKILL.md` | Model ID prefix table |
| `AGENTS.md` | Provider icons, prefix order (only if table lives there — prefer SKILL.md) |
| `README.md` | Provider list in config |
| `CHANGELOG.md` | |

### Frontend UX only (no API change)

| File | Action |
|------|--------|
| `AGENTS.md` | Component table, user flows |
| `README.md` | Screenshots / setup prose if misleading |
| `docs/QUICKSTART.md` | Step-by-step |
| `CHANGELOG.md` | |

### MCP-only (new/changed tools)

| File | Action |
|------|--------|
| `docs/mcp/TOOLS.md` | Full tool entry (params, examples) |
| `docs/mcp/INSTRUCTIONS.md` | Agent routing rules (mirror `llm_council_mcp/server.py`) |
| `llm_council_mcp/server.py` | MCP `instructions=` string — prefer tools over curl |
| `docs/mcp/README.md` | Tool count (`9`); `GET /api/health` → `mcp.tools` when total changes |
| `docs/mcp/EXAMPLES.md` | New workflows |
| `skills/llm-council-api/SKILL.md` | MCP-first routing + REST fallback table |
| `llm_council_mcp/tests/` | Tool tests |
| `CHANGELOG.md` | |

---

## Product rules (keep docs aligned)

Document these consistently everywhere they appear:

### Provider availability

| Surface | Model sources |
|---------|----------------|
| **LLM Council** (Settings → Council Config toggles) | `enabled_providers` + `direct_provider_toggles` filter which sources appear in **Settings** council pickers |
| **LLM Advisors** (Advisor Setup) | **All configured providers** (API keys + Ollama URL + custom endpoint). **Ignores** council `enabled_providers` toggles |
| **MCP / REST** | Use `GET /api/models`, `/api/models/direct`, `/api/ollama/tags`, `/api/custom-endpoint/models` — availability depends on keys, not council toggles |

### Settings vs main screen (planned / shipped)

| Data | Council main screen | Settings |
|------|---------------------|----------|
| `council_models`, `chairman_model` | Editable on welcome (Council Setup) **and** Settings — **same persisted fields**; locked in conversation after first message |
| `council_presets` | Welcome Council Setup UI + `PUT /api/settings` |
| `advisor_presets` | Advisor Setup UI + `PUT /api/settings` |
| Temperatures, prompts, provider toggles | Settings only |
| Lineup locked after first council message | Yes (v1) |

### Advisor presets (`advisor_presets`)

- Max 20 presets; one `is_default`
- Saves: personas, `mode` (simple/advanced), models, optional rounds + web search
- Does **not** save debate question text
- REST: `GET/PUT /api/settings` field `advisor_presets`
- MCP: `advisor_settings` action `get` returns presets; preset CRUD via `advisor_settings` preset actions (REST fallback: `PUT /api/settings`)

---

## Shipped in v0.5.2 (sync verified)

- [x] **MCP 9-tool consolidation** — `llm_council_mcp/tools/*.py`, `server.py`, `docs/mcp/TOOLS.md`, `INSTRUCTIONS.md`, SKILL routing table
- [x] **MCP-first routing** — SKILL MCP-first section, `server.py` instructions, `docs/mcp/INSTRUCTIONS.md`
- [x] **Inline council setup** — `CouncilSetup.jsx`, `council_presets`, auto-save, chairman optional in Chat Only
- [x] `advisor_presets` — backend, UI, SKILL §18, CHANGELOG, MCP `advisor_settings`
- [x] Advisors use configured providers (not council toggles) — SKILL, AGENTS, QUICKSTART, README, MCP TOOLS/EXAMPLES
- [x] Council Config toggles labeled council-only — UI copy + docs
- [x] `+ New Council` switches to council mode from advisors — CHANGELOG, AGENTS
- [x] `docs/DOC-SYNC.md` checklist — linked from README, AGENTS
- [x] MCP tests for 9-tool API (139 tests)

---

## PR self-check

Before merge, confirm:

- [ ] `CHANGELOG.md` `[Unreleased]` updated
- [ ] `skills/llm-council-api/SKILL.md` field tables and examples match `backend/settings.py`
- [ ] `docs/mcp/TOOLS.md` matches `llm_council_mcp/tools/*.py` signatures
- [ ] `AGENTS.md` reflects current UI flows (no stale “Settings only” for advisors)
- [ ] MCP tool JSON examples include new settings fields
- [ ] Tests updated for MCP GET settings shape
