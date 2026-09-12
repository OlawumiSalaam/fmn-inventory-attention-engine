"""Provider abstraction for the FMN Inventory Attention Engine LLM layer.

The client reads provider configuration from the project's root config.yaml
and provider credentials from environment variables. The deterministic
inventory application remains functional when no LLM provider is configured.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
import tomllib
from typing import Any

import yaml
from openai import OpenAI

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config.yaml"


class LLMUnavailableError(Exception):
    """Raised when no configured LLM provider can produce a response."""


@dataclass
class Provider:
    """Configured LLM provider and its OpenAI compatible client."""

    name: str
    model: str
    sdk: OpenAI
    temperature: float | None = None


@dataclass
class ChatReply:
    """Normalized response returned by an LLM provider."""

    content: str
    tool_calls: list[Any]
    provider: str
    model: str
    latency_ms: int


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    """Load the project's YAML configuration."""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def get_secret(name: str) -> str | None:
    """Return a provider API key from environment variables or secrets.toml."""
    value = os.environ.get(name)

    if value:
        return value

    secrets_path = PROJECT_ROOT / ".secrets" / "secrets.toml"

    if not secrets_path.exists():
        return None

    try:
        with secrets_path.open("rb") as handle:
            secrets = tomllib.load(handle)

        value = secrets.get(name)

        if value:
            return str(value)

    except (OSError, tomllib.TOMLDecodeError) as exc:
        logger.warning("Unable to read secrets file: %s", exc)

    return None


class LLMClient:
    """Try configured providers in order and expose one stable chat interface."""

    def __init__(self, providers: list[Provider]) -> None:
        """Initialise the client with at least one configured provider."""
        if not providers:
            raise LLMUnavailableError(
                "No LLM provider has an API key configured."
            )

        self.providers = providers

    @classmethod
    def from_config(
        cls,
        llm_config: dict[str, Any],
        only: str | None = None,
    ) -> "LLMClient":
        """Build a client from the application's LLM configuration."""
        providers: list[Provider] = []

        for settings in llm_config.get("providers", []):
            if only and settings["name"] != only:
                continue

            api_key = get_secret(settings["api_key_name"])

            if not api_key:
                logger.warning(
                    "No API key for %s, skipping provider.",
                    settings["name"],
                )
                continue

            providers.append(
                make_provider(
                    settings=settings,
                    api_key=api_key,
                    llm_config=llm_config,
                )
            )

        return cls(providers)

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
    ) -> ChatReply:
        """Generate a response, falling back to the next provider on failure."""
        errors: list[str] = []

        for provider in self.providers:
            request: dict[str, Any] = {
                "model": provider.model,
                "messages": messages,
            }

            if provider.temperature is not None:
                request["temperature"] = provider.temperature

            if tools:
                request["tools"] = tools

            start = time.perf_counter()

            try:
                response = provider.sdk.chat.completions.create(**request)

            except Exception as exc:
                logger.warning(
                    "%s (%s) failed: %s",
                    provider.name,
                    provider.model,
                    exc,
                )
                errors.append(
                    f"{provider.name}: {exc}"
                )
                continue

            latency_ms = int(
                (time.perf_counter() - start) * 1000
            )

            message = response.choices[0].message

            logger.info(
                "Reply from %s (%s) in %d ms",
                provider.name,
                provider.model,
                latency_ms,
            )

            return ChatReply(
                content=message.content or "",
                tool_calls=list(message.tool_calls or []),
                provider=provider.name,
                model=provider.model,
                latency_ms=latency_ms,
            )

        raise LLMUnavailableError(
            " | ".join(errors)
        )


def make_provider(
    settings: dict[str, Any],
    api_key: str,
    llm_config: dict[str, Any],
) -> Provider:
    """Create one provider from the application's provider configuration."""
    sdk = OpenAI(
        api_key=api_key,
        base_url=settings["base_url"],
        timeout=llm_config.get("timeout_seconds", 30),
        max_retries=llm_config.get("max_retries", 1),
    )

    return Provider(
        name=settings["name"],
        model=settings["model"],
        sdk=sdk,
        temperature=settings.get("temperature"),
    )