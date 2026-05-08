"""LM Studio provider (OpenAI-compatible local API)."""

import httpx
from typing import List, Dict, Any
from .base import LLMProvider
from ..settings import get_settings


class LmStudioProvider(LLMProvider):
    """Provider for LM Studio's local OpenAI-compatible API endpoint."""

    def _get_config(self) -> tuple[str, str]:
        """Get LM Studio endpoint configuration."""
        settings = get_settings()
        base_url = settings.lm_studio_base_url or "http://localhost:1234/v1"
        api_key = settings.lm_studio_api_key or ""
        return base_url, api_key

    async def query(self, model_id: str, messages: List[Dict[str, str]], timeout: float = 120.0, temperature: float = 0.7) -> Dict[str, Any]:
        base_url, api_key = self._get_config()

        # Strip prefix if present
        model = model_id.removeprefix("lmstudio:")

        # Normalize URL
        if base_url.endswith('/'):
            base_url = base_url[:-1]

        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=timeout) as client:
                response = await client.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": temperature
                    }
                )

                if response.status_code != 200:
                    return {
                        "error": True,
                        "error_message": f"LM Studio API error: {response.status_code} - {response.text}"
                    }

                data = response.json()
                content = data["choices"][0]["message"]["content"]
                return {"content": content, "error": False}

        except Exception as e:
            return {"error": True, "error_message": str(e)}

    async def get_models(self) -> List[Dict[str, Any]]:
        base_url, api_key = self._get_config()

        # Normalize URL
        if base_url.endswith('/'):
            base_url = base_url[:-1]

        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{base_url}/models",
                    headers=headers
                )

                if response.status_code != 200:
                    return []

                data = response.json()
                models = []

                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    if not model_id:
                        continue

                    mid = model_id.lower()
                    # Filter out non-chat models
                    if any(x in mid for x in ["embed", "whisper", "tts", "dall-e", "audio", "transcribe"]):
                        continue

                    models.append({
                        "id": f"lmstudio:{model_id}",
                        "name": f"{model_id} [LM Studio]",
                        "provider": "LM Studio",
                        "is_free": True
                    })

                return sorted(models, key=lambda x: x["name"])

        except Exception:
            return []

    async def validate_key(self, api_key: str) -> Dict[str, Any]:
        # For LM Studio, api_key is optional (local service)
        base_url, _ = self._get_config()
        return await self.validate_connection(base_url, api_key)

    async def validate_connection(self, url: str, api_key: str = "") -> Dict[str, Any]:
        """Validate connection to LM Studio endpoint."""
        if not url:
            return {"success": False, "message": "URL is required"}

        # Normalize URL
        if url.endswith('/'):
            url = url[:-1]

        try:
            headers = {}
            if api_key:
                headers["Authorization"] = f"Bearer {api_key}"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    f"{url}/models",
                    headers=headers
                )

                if response.status_code == 200:
                    data = response.json()
                    model_count = len(data.get("data", []))
                    return {
                        "success": True,
                        "message": f"Connected to LM Studio. Found {model_count} models."
                    }
                elif response.status_code == 401:
                    return {"success": False, "message": "Authentication failed. Check your API key."}
                else:
                    return {"success": False, "message": f"API error: {response.status_code}"}

        except httpx.ConnectError:
            return {"success": False, "message": "Connection failed. Check the URL. Is LM Studio running?"}
        except httpx.TimeoutException:
            return {"success": False, "message": "Connection timed out. Check the URL."}
        except Exception as e:
            return {"success": False, "message": str(e)}
