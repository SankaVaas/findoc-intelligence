"""
LLM client — wraps Ollama with plain chat + structured JSON output.

Works with any Ollama model: qwen2.5:0.5b (dev), phi3:mini, mistral:7b (prod).
Switching models is just changing LLM_MODEL in .env — zero code changes.

Usage:
    llm = LLMClient()
    text   = llm.chat("What is 2+2?")
    result = llm.structured("Extract the company name", MySchema)
"""

from __future__ import annotations

import json
from typing import Type

from pydantic import BaseModel
from utils.logging import logger
from utils.settings import settings


class LLMClient:

    def __init__(
        self,
        model      : str | None = None,
        base_url   : str | None = None,
        temperature: float = 0.1,
        max_tokens : int   = 1024,
    ):
        self.model       = model    or settings.llm_model
        self.base_url    = (base_url or settings.ollama_base_url).rstrip("/")
        self.temperature = temperature
        self.max_tokens  = max_tokens

    # ── Sync chat ─────────────────────────────────────────────────────────

    def chat(self, user: str, system: str = "") -> str:
        """Plain text generation."""
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat",
                json={
                    "model"   : self.model,
                    "messages": messages,
                    "stream"  : False,
                    "options" : {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=120.0,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except Exception as e:
            logger.error("LLM chat failed: {}", e)
            raise

    def structured(
        self,
        user   : str,
        schema : Type[BaseModel],
        system : str = "",
        retries: int = 2,
    ) -> BaseModel:
        """
        Generate a response conforming to `schema`.
        Injects schema into system prompt and retries on parse failure.
        """
        json_system = (
            f"{system}\n\n"
            f"You MUST respond with valid JSON matching this schema exactly:\n"
            f"{json.dumps(schema.model_json_schema(), indent=2)}\n"
            f"Return ONLY the JSON object. No explanation. No markdown fences."
        ).strip()

        for attempt in range(retries + 1):
            raw = self.chat(user, system=json_system)
            try:
                clean = (
                    raw.strip()
                    .removeprefix("```json")
                    .removeprefix("```")
                    .removesuffix("```")
                    .strip()
                )
                # Extract first JSON object if model added extra text
                start = clean.find("{")
                end   = clean.rfind("}") + 1
                if start != -1 and end > start:
                    clean = clean[start:end]
                return schema.model_validate(json.loads(clean))
            except Exception as e:
                if attempt < retries:
                    logger.debug("Structured output retry {}/{}: {}", attempt + 1, retries, e)
                    user = f"{user}\n\nYour previous response was invalid JSON: {e}. Fix it."
                else:
                    logger.error("Structured output failed after {} retries: {}", retries, e)
                    raise

    # ── Async streaming ───────────────────────────────────────────────────

    async def astream(self, user: str, system: str = ""):
        """Async token streaming for the FastAPI /query/stream endpoint."""
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        async with httpx.AsyncClient(timeout=120.0) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model"   : self.model,
                    "messages": messages,
                    "stream"  : True,
                    "options" : {"temperature": self.temperature},
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if line:
                        try:
                            token = json.loads(line).get("message", {}).get("content", "")
                            if token:
                                yield token
                        except json.JSONDecodeError:
                            continue

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        import httpx
        try:
            resp   = httpx.get(f"{self.base_url}/api/tags", timeout=5.0)
            models = [m["name"] for m in resp.json().get("models", [])]
            ok     = any(self.model.split(":")[0] in m for m in models)
            if not ok:
                logger.warning(
                    "Model '{}' not found. Available: {}. "
                    "Run: ollama pull {}",
                    self.model, models, self.model,
                )
            return ok
        except Exception as e:
            logger.warning("Ollama not reachable at {}: {}", self.base_url, e)
            return False