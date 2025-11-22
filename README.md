# LLM Council

![llmcouncil](header.jpg)

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT 5.1, Google Gemini 3.0 Pro, Anthropic Claude Sonnet 4.5, xAI Grok 4), you can group them into your "LLM Council".

This is a fully local, lightweight web app. The code has to be minimal, readable, simple and not bloated with complexity. The idea is that the user runs the web app locally, and access it via a localhost url/port.

UIUX wise, the app looks like ChatGPT. Meaning there is a sidebar on the left storing conversations. User can create a new conversation or access past conversations. The conversations persist.

The life of a single conversation looks as follows. The user writes up a query. Then:

1. Stage 1: first opinions. The user query is given to all LLMs individually, and the responses are collected. The individual responses are shown in a "tab view", so that the user can inspect them.
2. Stage 2: review. Each individual LLM is given the responses of the other LLMs. The LLM identities are anonymized so that the LLM can't play favorites when judging their outputs. The LLM is asked to rank them in accuracy and insight.
3. Stage 3: final response. The designated Chairman of the LLM Council (the strongest model) takes all of the model's responses and compiles them into a single final answer that is presented to the user.

Implementation details:

- The project uses [uv](https://docs.astral.sh/uv/) for project management.
- The project uses [OpenRouter](https://openrouter.ai/) to easily call models across all API providers with a single API key. The API key is stored in `.env` file in current repo.

## Setup

### 1. Install Dependencies

**Backend:**
```bash
uv sync
```

**Frontend:**
```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API key at [openrouter.ai](https://openrouter.ai/). Make sure to purchase the credits you need, or sign up for automatic top up.

### 3. Configure Models (Optional)

Edit `backend/config.py` to customize the council:

```python
COUNCIL_MODELS = [
    "openai/gpt-5.1",
    "google/gemini-3-pro-preview",
    "anthropic/claude-sonnet-4.5",
    "x-ai/grok-4",
]

CHAIRMAN_MODEL = "google/gemini-3-pro-preview"
```

## Running the Application

**Option 1: Use the start script**
```bash
./start.sh
```

**Option 2: Run manually**

Terminal 1 (Backend):
```bash
uv run python -m backend.main
```

Terminal 2 (Frontend):
```bash
cd frontend
npm run dev
```

Then open http://localhost:5173 in your browser.

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/`
- **Package Management:** uv for Python, npm for JavaScript