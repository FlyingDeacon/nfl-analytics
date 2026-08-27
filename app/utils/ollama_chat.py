"""Thin client for an Ollama server — a local one in development, Ollama Cloud
once the app is deployed.

Both speak the same native /api/chat, so the only real difference is the host
and a bearer token. Config resolves in this order, first hit winning:

  1. st.secrets["ollama"] — how Streamlit Cloud gets the hosted endpoint + key
  2. OLLAMA_HOST / OLLAMA_API_KEY / OLLAMA_MODEL environment variables
  3. a plain local server on http://localhost:11434

so a laptop with `ollama serve` running needs no configuration at all. See
.streamlit/secrets.toml for the deployed shape:

    [ollama]
    host  = "https://ollama.com"
    key   = "..."            # ollama.com -> Settings -> API keys
    model = "gpt-oss:20b"

Everything here is best-effort: when nothing is reachable the helpers return
empty rather than raising, so a page can degrade to a notice instead of
blowing up.
"""

from __future__ import annotations

import json
import os
from typing import Iterator

import requests
import streamlit as st

LOCAL_HOST = "http://localhost:11434"
FALLBACK_MODEL = "llama3.1:8b"

PROBE_TIMEOUT = 4      # seconds — just checking the endpoint answers
CHAT_TIMEOUT = 180     # seconds — local generation on CPU can be slow


def _secrets() -> dict:
    """The [ollama] block, or {} when the app has no secrets file at all."""
    try:
        return dict(st.secrets.get("ollama", {}))
    except Exception:  # no secrets.toml on disk — the normal local-dev case
        return {}


def config() -> tuple[str, str, str]:
    """Resolved (host, api_key, preferred_model)."""
    sec = _secrets()
    host = sec.get("host") or os.environ.get("OLLAMA_HOST") or LOCAL_HOST
    key = sec.get("key") or os.environ.get("OLLAMA_API_KEY") or ""
    model = sec.get("model") or os.environ.get("OLLAMA_MODEL") or FALLBACK_MODEL
    return host.rstrip("/"), key, model


def _headers(key: str) -> dict:
    return {"Authorization": f"Bearer {key}"} if key else {}


def list_models() -> list[str]:
    """Models we can offer, preferred first. Empty means nothing is reachable.

    A local server answers /api/tags with whatever is pulled. Ollama Cloud
    doesn't necessarily advertise its catalogue that way, so a keyed host falls
    back to the single configured model rather than reporting itself offline.
    """
    host, key, preferred = config()
    try:
        resp = requests.get(
            f"{host}/api/tags", headers=_headers(key), timeout=PROBE_TIMEOUT
        )
        resp.raise_for_status()
        names = [m["name"] for m in resp.json().get("models", [])]
    except (requests.RequestException, ValueError, KeyError, TypeError):
        names = []

    if not names:
        return [preferred] if key else []
    return sorted(names, key=lambda n: (n != preferred, n))


def stream_chat(model: str, messages: list[dict], temperature: float = 0.2) -> Iterator[str]:
    """Yield the reply to `messages` token by token from Ollama's /api/chat."""
    host, key, _ = config()
    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": temperature},
    }
    with requests.post(
        f"{host}/api/chat",
        json=payload,
        headers=_headers(key),
        stream=True,
        timeout=CHAT_TIMEOUT,
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
