"""Thin client for a locally running Ollama server.

Everything here is best-effort: if Ollama isn't running the helpers return
empty/None rather than raising, so a page can degrade to a quiet notice instead
of blowing up.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import requests

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
PREFERRED_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")

PROBE_TIMEOUT = 2      # seconds — just checking the server is up
CHAT_TIMEOUT = 180     # seconds — local generation on CPU can be slow


def list_models() -> list[str]:
    """Models the local server can serve, best first. Empty when it isn't up."""
    try:
        resp = requests.get(f"{HOST}/api/tags", timeout=PROBE_TIMEOUT)
        resp.raise_for_status()
        names = sorted(m["name"] for m in resp.json().get("models", []))
    except (requests.RequestException, ValueError, KeyError):
        return []
    # Float the configured favourite to the top so it's the default selection.
    return sorted(names, key=lambda n: (n != PREFERRED_MODEL, n))


def stream_chat(model: str, messages: list[dict], temperature: float = 0.2) -> Iterator[str]:
    """Yield the reply to `messages` token by token from Ollama's /api/chat."""
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    with requests.post(
        f"{HOST}/api/chat", json=payload, stream=True, timeout=CHAT_TIMEOUT
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except ValueError:
                continue
            if chunk.get("error"):
                raise RuntimeError(chunk["error"])
            piece = chunk.get("message", {}).get("content", "")
            if piece:
                yield piece
            if chunk.get("done"):
                break
