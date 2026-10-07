"""
Concrete adapters for the `Generator` port.

  fake  -> _FakeGenerator   -> CI / unit tests. Deterministic, offline, no deps.
  openai -> _OpenAIBackend   -> any OpenAI-compatible /v1/chat/completions server.
                      vLLM (prod, Qwen2.5-7B-AWQ), Ollama (dev, no GPU),
                      TGI / llama.cpp / Azure OpenAI — only base_url + model
                      differ; the adapter code is identical.
"""

from __future__ import annotations

import logging
from typing import Optional

import openai as _openai

from ragapp.exceptions.domain import GeneratorError

logger = logging.getLogger(__name__)


class _FakeGenerator:
    """Deterministic, offline generator for tests and GPU-free environments."""

    def __init__(self, canned: Optional[str] = None) -> None:
        self._canned = canned
        self.closed = False

    def generate(self, question: str) -> str:
        if self._canned is not None:
            return self._canned
        last_user = next(
            (m["content"] for m in reversed(question) if m.get("role") == "user"),
            "",
        )
        return f"[fake-generator] {last_user}"

    def close(self) -> None:
        self.closed = True


class _OpenAIBackend:
    """One OpenAI-compatible client; the backend is configuration."""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        timeout: float = 120.0,
    ) -> None:

        # self._openai = _openai
        self._model = model
        self._temperature = float(temperature)
        self._max_tokens = max_tokens
        self._client = _openai.OpenAI(
            base_url=base_url,
            api_key=api_key or "EMPTY",
            timeout=timeout,
        )

    def generate(self, question: list[dict[str, str]]) -> str:
        kwargs: dict = {
            "model": self._model,
            "messages": question,
            "temperature": self._temperature,
        }

        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.warning(
                "generator_upstream_error",
                extra={
                    "http": {"status": 502},
                    "model": self._model,
                    "error": f"{type(e).__name__}: {e}",
                },
            )
            raise GeneratorError(
                message="Generator upstream error", details={"model": self._model}
            ) from e

        if not resp.choices:
            raise GeneratorError(
                message="Generator returned no choices",
                details={"model": self._model},
            )
        return resp.choices[0].message.content or ""

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass
