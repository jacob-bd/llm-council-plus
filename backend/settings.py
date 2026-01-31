"""Settings storage and management."""

import json
import os
from pathlib import Path
from typing import Optional, List, Dict
from pydantic import BaseModel
from .search import SearchProvider

# Settings file path
SETTINGS_FILE = Path(__file__).parent.parent / "data" / "settings.json"

# Default models (matches original llm-council defaults)
DEFAULT_COUNCIL_MODELS = ["", ""]
DEFAULT_CHAIRMAN_MODEL = ""

# Default enabled providers
DEFAULT_ENABLED_PROVIDERS = {
    "openrouter": True,
    "ollama": False,
    "groq": False,
    "direct": False,  # Master toggle for all direct connections
    "custom": False   # Custom OpenAI-compatible endpoint
}

# Default direct provider toggles (individual)
DEFAULT_DIRECT_PROVIDER_TOGGLES = {
    "openai": False,
    "anthropic": False,
    "google": False,
    "mistral": False,
    "deepseek": False,
    "groq": False
}


# Available models for selection (popular OpenRouter models)
AVAILABLE_MODELS = [
    # OpenAI
    {"id": "openai/gpt-4o", "name": "GPT-4o [OpenRouter]", "provider": "OpenAI", "source": "openrouter"},
    {"id": "openai/gpt-4o-mini", "name": "GPT-4o Mini [OpenRouter]", "provider": "OpenAI", "source": "openrouter"},
    {"id": "openai/o1-preview", "name": "o1 Preview [OpenRouter]", "provider": "OpenAI", "source": "openrouter"},
    {"id": "openai/o1-mini", "name": "o1 Mini [OpenRouter]", "provider": "OpenAI", "source": "openrouter"},
    # Google
    {"id": "google/gemini-pro-1.5", "name": "Gemini 1.5 Pro [OpenRouter]", "provider": "Google", "source": "openrouter", "is_free": True},
    {"id": "google/gemini-flash-1.5", "name": "Gemini 1.5 Flash [OpenRouter]", "provider": "Google", "source": "openrouter", "is_free": True},
    {"id": "google/gemini-pro-vision", "name": "Gemini Pro Vision [OpenRouter]", "provider": "Google", "source": "openrouter"},
    # Anthropic
    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet [OpenRouter]", "provider": "Anthropic", "source": "openrouter"},
    {"id": "anthropic/claude-3-opus", "name": "Claude 3 Opus [OpenRouter]", "provider": "Anthropic", "source": "openrouter"},
    {"id": "anthropic/claude-3-haiku", "name": "Claude 3 Haiku [OpenRouter]", "provider": "Anthropic", "source": "openrouter"},
    # Meta
    {"id": "meta-llama/llama-3.1-405b-instruct", "name": "Llama 3.1 405B [OpenRouter]", "provider": "Meta", "source": "openrouter"},
    {"id": "meta-llama/llama-3.1-70b-instruct", "name": "Llama 3.1 70B [OpenRouter]", "provider": "Meta", "source": "openrouter", "is_free": True},
    # Mistral
    {"id": "mistralai/mistral-large", "name": "Mistral Large [OpenRouter]", "provider": "Mistral", "source": "openrouter"},
    {"id": "mistralai/mistral-medium", "name": "Mistral Medium [OpenRouter]", "provider": "Mistral", "source": "openrouter"},
    # DeepSeek
    {"id": "deepseek/deepseek-chat", "name": "DeepSeek V3 [OpenRouter]", "provider": "DeepSeek", "source": "openrouter"},
]


from .prompts import (
    STAGE1_PROMPT_DEFAULT,
    STAGE2_PROMPT_DEFAULT,
    STAGE3_PROMPT_DEFAULT
)

class Settings(BaseModel):
    """Application settings."""
    search_provider: SearchProvider = SearchProvider.DUCKDUCKGO
    search_keyword_extraction: str = "direct"  # "direct" or "yake"
    search_result_count: int = 8  # Number of search results (5-15, default 8)
    search_hybrid_mode: bool = True  # Combine web+news search for DuckDuckGo

    # API Keys
    tavily_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    openrouter_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    mistral_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None

    # Ollama Settings
    ollama_base_url: str = "http://localhost:11434"

    # Custom OpenAI-compatible endpoint
    custom_endpoint_name: Optional[str] = None
    custom_endpoint_url: Optional[str] = None
    custom_endpoint_api_key: Optional[str] = None

    # Enabled Providers (which sources are available for council selection)
    enabled_providers: Dict[str, bool] = DEFAULT_ENABLED_PROVIDERS.copy()

    # Individual direct provider toggles
    direct_provider_toggles: Dict[str, bool] = DEFAULT_DIRECT_PROVIDER_TOGGLES.copy()

    # Council Configuration (unified across all providers)
    council_models: List[str] = DEFAULT_COUNCIL_MODELS.copy()
    chairman_model: str = DEFAULT_CHAIRMAN_MODEL
    
    # Temperature Settings
    council_temperature: float = 0.5
    chairman_temperature: float = 0.4
    stage2_temperature: float = 0.3  # Lower for consistent ranking output
    
    # Remote/Local filters
    council_member_filters: Optional[Dict[int, str]] = None
    chairman_filter: Optional[str] = None
    search_query_filter: Optional[str] = None

    full_content_results: int = 3  # Number of search results to fetch full content for (0 to disable)
    show_free_only: bool = False  # Filter to show only free OpenRouter models

    # System Prompts
    stage1_prompt: str = STAGE1_PROMPT_DEFAULT
    stage2_prompt: str = STAGE2_PROMPT_DEFAULT
    stage3_prompt: str = STAGE3_PROMPT_DEFAULT
    
    # Execution Mode
    execution_mode: str = "full"  # Default execution mode: 'chat_only', 'chat_ranking', 'full'


def get_settings() -> Settings:
    """Load settings from file, or return defaults."""
    if SETTINGS_FILE.exists():
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
                return Settings(**data)
        except Exception:
            pass
    return Settings()


def save_settings(settings: Settings) -> None:
    """Save settings to file."""
    # Ensure data directory exists
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(SETTINGS_FILE, "w") as f:
        json.dump(settings.model_dump(), f, indent=2)


def update_settings(**kwargs) -> Settings:
    """Update specific settings and save."""
    current = get_settings()
    updated_data = current.model_dump()
    updated_data.update(kwargs)
    updated = Settings(**updated_data)
    save_settings(updated)
    return updated
