"""Minimal client for an OpenAI-compatible chat endpoint (LM Studio, llama.cpp, vLLM, Ollama)."""

from __future__ import annotations

import re

import httpx

from app.config import Settings

# Qwen3 emits its reasoning in <think> blocks; strip it from the user-facing answer.
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, settings: Settings):
        self.base_url = settings.lm_base_url.rstrip("/")
        self.model = settings.lm_model
        self.api_key = settings.lm_api_key
        self.timeout = settings.lm_timeout
        self.thinking = settings.thinking

    def chat(self, messages: list[dict], temperature: float = 0.2, max_tokens: int = 1024) -> str:
        body = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if not self.thinking:
            # Ask for Qwen3's non-thinking mode, the format the fine-tuned model was trained on: its prompt
            # must end in an empty <think></think>. A fine-tuned model shown the thinking-mode prompt instead
            # can emit junk before answering. LM Studio reads reasoning_effort; llama.cpp's server reads
            # chat_template_kwargs. Servers ignore the one they don't know.
            body["reasoning_effort"] = "none"
            body["chat_template_kwargs"] = {"enable_thinking": False}
        try:
            r = httpx.post(
                f"{self.base_url}/chat/completions",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=body,
                timeout=self.timeout,
            )
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise LLMError(f"LLM request to {self.base_url} failed: {e}") from e
        content = r.json()["choices"][0]["message"]["content"] or ""
        return _THINK.sub("", content).strip()

    def models(self) -> list[str]:
        r = httpx.get(f"{self.base_url}/models", headers={"Authorization": f"Bearer {self.api_key}"}, timeout=10)
        r.raise_for_status()
        return [m["id"] for m in r.json().get("data", [])]
