"""
clients.py — Centralized LLM client singletons.

All LLM clients in this codebase should be obtained via these getters rather
than instantiated directly. Benefits:
  - One client instance per backend (no redundant re-auth)
  - Tracing instrumentation (Phoenix) is guaranteed to be applied before any
    client is built, because every getter is lazy (first-call only)
  - URLs and env-var names live in one place

Usage:
    from litesr.clients import get_anthropic_client, get_ollama_client
    from litesr.clients import get_openai_client, get_hf_client

    client = get_anthropic_client()
    client = get_ollama_client()                         # default: localhost:11434
    client = get_ollama_client("http://myhost:11434/v1") # custom URL
    client = get_openai_client()                         # reads API_KEY env var
    client = get_hf_client()                             # reads HF_TOKEN env var
"""

from __future__ import annotations

import os
from typing import Optional

# ── Optional deps ──────────────────────────────────────────────────────────────
try:
    import anthropic as _anthropic_lib
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

try:
    from openai import OpenAI as _OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


# ── Default URLs (override via env vars) ──────────────────────────────────────
OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
HF_BASE_URL: str = os.environ.get("HF_API_URL", "https://router.huggingface.co/v1")


# ── Singletons ─────────────────────────────────────────────────────────────────
_anthropic_client: Optional[object] = None
_ollama_client: Optional[object] = None
_ollama_client_url: Optional[str] = None   # track which URL the singleton was built for
_openai_client: Optional[object] = None


def get_anthropic_client():
    """
    Return a shared anthropic.Anthropic() instance.

    Reads ANTHROPIC_API_KEY from the environment (standard SDK behaviour).
    The client is created on first call so that any Phoenix instrumentation
    applied before the first call is captured correctly.
    """
    global _anthropic_client
    if not _ANTHROPIC_AVAILABLE:
        raise ImportError("anthropic package is required: pip install anthropic")
    if _anthropic_client is None:
        _anthropic_client = _anthropic_lib.Anthropic()
    return _anthropic_client


def get_ollama_client(base_url: str = OLLAMA_BASE_URL):
    """
    Return a shared OpenAI-compatible client pointed at a local Ollama server.

    The singleton is rebuilt if a different base_url is requested.
    """
    global _ollama_client, _ollama_client_url
    if not _OPENAI_AVAILABLE:
        raise ImportError("openai package is required: pip install openai")
    if _ollama_client is None or _ollama_client_url != base_url:
        _ollama_client = _OpenAI(api_key="ollama", base_url=base_url)
        _ollama_client_url = base_url
    return _ollama_client


def get_openai_client(api_key: str | None = None):
    """
    Return a shared OpenAI client.

    Reads the API_KEY environment variable if api_key is not provided.
    The client is created on first call.
    """
    global _openai_client
    if not _OPENAI_AVAILABLE:
        raise ImportError("openai package is required: pip install openai")
    if _openai_client is None:
        key = api_key or os.environ.get("API_KEY")
        if not key:
            raise EnvironmentError("OpenAI API key not found. Set API_KEY env var or pass api_key=.")
        _openai_client = _OpenAI(api_key=key)
    return _openai_client


def llm_call(
    prompt: str,
    max_tokens: int,
    use_ollama: bool = False,
    ollama_model: str = "mistral:latest",
    anthropic_model: str = "claude-sonnet-4-6",
) -> str:
    """Call either Ollama or Anthropic and return the response text."""
    if use_ollama:
        client = get_ollama_client()
        resp = client.chat.completions.create(
            model=ollama_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content.strip()
    else:
        client = get_anthropic_client()
        resp = client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text.strip()


def llm_call_vision(
    prompt: str,
    image_b64: str,
    max_tokens: int = 1000,
    use_ollama: bool = False,
    ollama_model: str = "llava:latest",
    anthropic_model: str = "claude-sonnet-4-6",
) -> str:
    """Call a vision-capable model with a base64 PNG image and return the response text."""
    if use_ollama:
        client = get_ollama_client()
        resp = client.chat.completions.create(
            model=ollama_model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content.strip()
    else:
        client = get_anthropic_client()
        resp = client.messages.create(
            model=anthropic_model,
            max_tokens=max_tokens,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        return resp.content[0].text.strip()


def llm_call_pdf_document(
    pdf_b64:         str,
    prompt:          str,
    max_tokens:      int = 4096,
    anthropic_model: str = "claude-sonnet-4-6",
) -> str:
    """Send a base64-encoded PDF chunk to Claude and return the response text.
    Uses streaming to avoid the 10-minute timeout on large documents.
    """
    client = get_anthropic_client()

    with client.messages.stream(
        model      = anthropic_model,
        max_tokens = max_tokens,
        messages   = [{
            "role": "user",
            "content": [
                {"type": "document", "source": {
                    "type":       "base64",
                    "media_type": "application/pdf",
                    "data":       pdf_b64,
                }},
                {"type": "text", "text": prompt},
            ],
        }],
        extra_headers = {"anthropic-beta": "pdfs-2024-09-25"},
    ) as stream:
        return stream.get_final_text().strip()

def get_hf_client(api_key: str | None = None, base_url: str = HF_BASE_URL):
    """
    Return an OpenAI-compatible client pointed at the HuggingFace Inference Router.

    Not cached as a singleton because base_url can vary per config.
    Reads HF_TOKEN from the environment if api_key is not provided.
    """
    if not _OPENAI_AVAILABLE:
        raise ImportError("openai package is required: pip install openai")
    key = api_key or os.environ.get("HF_TOKEN")
    if not key:
        raise EnvironmentError("HuggingFace token not found. Set HF_TOKEN env var or pass api_key=.")
    return _OpenAI(api_key=key, base_url=base_url)