# Quick Start Guide

Get LLM Council Plus running in under 5 minutes.

---

## 1. Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **[uv](https://docs.astral.sh/uv/)** - Install with: `curl -LsSf https://astral.sh/uv/install.sh | sh`

---

## 2. Install & Run

```bash
# Clone the repo
git clone https://github.com/jacob-bd/llm-council-plus.git
cd llm-council-plus

# Install dependencies
uv sync
npm install --prefix frontend

# Start the app
./start.sh
```

Open **http://localhost:5173** in your browser.

---

## 3. First-Time Setup

The Settings panel opens automatically on first launch.

### Option A: Use OpenRouter (Easiest)
1. Get a free API key at [openrouter.ai/keys](https://openrouter.ai/keys)
2. Paste it in **LLM API Keys** → **OpenRouter**
3. Click **Test** (auto-saves on success)
4. Go to **Council Config** → Select models for your council
5. Click **Save Changes**

### Option B: Use Ollama (Free & Local)
1. Install [Ollama](https://ollama.com/)
2. Pull a model: `ollama pull llama3.1`
3. Start Ollama: `ollama serve`
4. In Settings → **LLM API Keys** → Click **Connect** for Ollama
5. Go to **Council Config** → Enable "Local (Ollama)" → Select models for your **council**
6. Click **Save Changes**

> **Advisors:** Once Ollama is connected in LLM API Keys, Advisor Setup lists Ollama models automatically — you do **not** need to enable the council "Local (Ollama)" toggle for advisors.

### Option C: Use Direct APIs
1. Get API keys from your preferred providers (OpenAI, Anthropic, Google, etc.)
2. Enter keys in **LLM API Keys** → **Direct LLM Connections**
3. Click **Test** for each (auto-saves on success)
4. Go to **Council Config** → Enable "Direct Connections" → Select models
5. Click **Save Changes**

---

## 4. Your First Council Query

1. Click **+ New Council** in the sidebar (or use the welcome screen)
2. **Configure your council** on the welcome screen — add members, pick a chairman (Full Deliberation only), or load a preset
3. Type your question
4. (Optional) Toggle **Web Search** for real-time grounding
5. Press **Enter**

Watch as:
- **Stage 1**: Each council member responds independently
- **Stage 2**: Models anonymously rank each other's responses
- **Stage 3**: Chairman synthesizes the final answer

---

## 4b. Your First Advisor Debate

1. Click **+ New Advisors** in the sidebar
2. Type a question or a decision to debate (e.g., "Should we rewrite our backend in Go?")
3. Configure the debate options:
   - Select 2 to 4 advisor personas (Skeptic, Strategist, Ethicist, etc.)
   - Set the number of back-and-forth rounds (3 to 10)
   - Choose a default model or assign specific models to individual personas — **all configured providers** appear here (API keys + Ollama + custom endpoint), not just council-enabled toggles
   - *(Optional)* Save your lineup as a **preset** from Model Assignment (personas, models, rounds, web search — not the debate question)
4. Click **Start Debate** and watch the advisors debate each other by name, culminating in a structured consensus verdict with a recommended action plan!

---

## 5. Deliberation Modes

Choose your deliberation type and depth:

| Mode / System | What Happens | Best For |
|------|--------------|----------|
| **Chat Only (Council)** | Just Stage 1 — quick individual responses | Quick model comparisons |
| **Chat + Ranking (Council)** | Stages 1 & 2 — see peer rankings and scores | Peer review without synthesis |
| **Full Deliberation (Council)** | All 3 stages — complete synthesis (default) | Broad, highly accurate synthesis |
| **LLM Advisors (Advisory)** | Persona-driven debate across configurable rounds | Complex strategic/moral decisions |

---

## 6. Quick Tips

- **Mix model families** in the council for diverse perspectives (e.g., GPT + Claude + Gemini)
- **Assign specific models to personas**: Give *The Skeptic* a highly detailed model (like Claude) and *The Pragmatist* a fast model (like Groq)
- **Advisor presets**: Save recurring panels (e.g., "Startup Review") from Advisor Setup → Model Assignment
- **Use Groq** for ultra-fast council inference
- **Use Ollama** for unlimited, free local queries (great for local Chairman synthesis using a model like `granite4:1b`)
- **"I'm Feeling Lucky"** randomizes your council composition
- **Customize Personas**: Go to **Settings** → **Advisors** to edit name, description, emoji, and prompt for any advisor persona
- **Abort anytime** with the stop button in the sidebar

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Council models not appearing | Enable the provider in **Council Config** toggles; verify API key / Ollama connection |
| Advisor models not appearing | Configure **LLM API Keys** (keys + Ollama URL + custom endpoint). Council toggles do not apply to advisors |
| Rate limit errors | Use Groq (14k/day) or Ollama (unlimited) |
| Port conflict | Backend uses 8001, frontend uses 5173 |
| node_modules errors | `rm -rf frontend/node_modules && npm install --prefix frontend` |

---

## Next Steps

- Explore **System Prompts** to customize model behavior
- Configure **Web Search** providers (Tavily, Brave) for better results
- Adjust **Temperature** sliders for creativity control
- **Export** your council config to share or backup
- **Enable multi-round debate** — see [Council Debate Config Guide](COUNCIL-DEBATE-CONFIG.md) for a full walkthrough of critique modes, round settings, and real-world examples

For full documentation, see [README.md](README.md).

---

<p align="center">
  <em>Ask the council. Get better answers.</em>
</p>
