"""
llm_client.py — Thin LLM wrapper used by all pipeline stages.

WHAT IT DOES:
  Provides a single `complete(prompt, system)` function that:
    1. Calls the OpenAI-compatible API (or any LiteLLM provider)
    2. Returns the response text as a plain string
    3. Optionally parses JSON from the response

WHY A WRAPPER:
  Every stage (TOC detection, tree building, summarization,
  retrieval) needs to call an LLM.  By centralising the call here,
  we can:
    - Swap models in one place (gpt-4o → claude-3 → local llama)
    - Add retry logic, rate limiting, and logging once
    - Test each stage by mocking this module

LEARNING FOCUS:
  - Token budget awareness: different stages need different max_tokens
  - JSON extraction: many stages expect structured JSON output
  - Why "fail loudly" is important (malformed JSON is a real problem)
"""

import json
import os
import re
import time
from typing import Optional, Union


# ─────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────

DEFAULT_MODEL = os.getenv("PAGEINDEX_MODEL", "gpt-4o")
# For Anthropic: "claude-opus-4-5" or "claude-sonnet-4-5"
# For local:     "ollama/llama3" (requires LiteLLM)


# ─────────────────────────────────────────────────
# Core completion function
# ─────────────────────────────────────────────────

def complete(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    retries: int = 3,
) -> str:
    """
    Call the LLM and return the response as a string.

    DESIGN NOTE (temperature=0.0):
      PageIndex uses temperature 0 everywhere.  Tree building and
      retrieval require *deterministic* structural outputs — we want
      the model to follow the schema, not be creative.  Generation
      (Step 8) can use a slightly higher temperature if desired.
    """
    client = _get_client()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"[llm_client] Retry {attempt + 1} after {wait}s: {e}")
            time.sleep(wait)

    raise RuntimeError("LLM completion failed after retries")


def complete_json(
    prompt: str,
    system: str = "",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    retries: int = 3,
) -> Union[dict, list]:
    """
    Call the LLM and parse the response as JSON.

    DESIGN NOTE:
      PageIndex prompts all use "respond ONLY with valid JSON".
      Even so, LLMs sometimes wrap JSON in ```json ... ``` fences.
      We strip those before parsing.

    Raises:
      json.JSONDecodeError if parsing fails after stripping fences
    """
    raw = complete(prompt, system=system, model=model,
                   max_tokens=max_tokens, retries=retries)
    return _extract_json(raw)


# ─────────────────────────────────────────────────
# JSON extraction helpers
# ─────────────────────────────────────────────────

def _extract_json(text: str) -> Union[dict, list]:
    """
    Extract and parse JSON from an LLM response.

    Strategy (in order of preference):
      1. Try parsing the text directly
      2. Strip markdown ```json fences, then parse
      3. Find the first {...} or [...] block, then parse
    """
    # Strategy 1: direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strategy 2: strip fences
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip())
    stripped = re.sub(r"\s*```$", "", stripped)
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass

    # Strategy 3: find first JSON block
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError(
        "Could not extract JSON from LLM response",
        text, 0
    )


# ─────────────────────────────────────────────────
# Client factory
# ─────────────────────────────────────────────────

_client_instance = None

def _get_client():
    """
    Lazy-initialise the OpenAI client.

    DESIGN NOTE:
      We use the openai SDK as the universal client because:
        - OpenAI, Anthropic (via proxy), and local models (Ollama)
          all speak the OpenAI-compatible chat completions API
        - LiteLLM can sit in front of any of them
      This means the pipeline itself never needs to know which
      provider it's talking to.
    """
    global _client_instance
    if _client_instance is None:
        try:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY", "not-set")
            base_url = os.getenv("OPENAI_BASE_URL", None)
            kwargs = {"api_key": api_key}
            if base_url:
                kwargs["base_url"] = base_url
            _client_instance = OpenAI(**kwargs)
        except ImportError:
            raise ImportError(
                "openai package required: pip install openai"
            )
    return _client_instance


# ─────────────────────────────────────────────────
# Mock client for unit testing (no API key needed)
# ─────────────────────────────────────────────────

class MockLLMClient:
    """
    Drop-in mock for testing pipeline stages without real API calls.

    Usage:
        import pageindex_scratch.llm_client as llm
        llm._client_instance = MockLLMClient(responses={...})
    """

    def __init__(self, responses: Optional[dict] = None):
        """
        responses: dict mapping prompt substrings to mock return values
        """
        self.responses = responses or {}
        self.calls: list[str] = []

    def chat_complete(self, prompt: str) -> str:
        self.calls.append(prompt)
        for key, value in self.responses.items():
            if key.lower() in prompt.lower():
                return value
        return '{"nodes": [], "title": "Mock Section", "start_index": 0, "end_index": 1}'


def complete_with_mock(prompt: str, mock_response: str) -> str:
    """
    Call complete() but inject a fixed response.
    Useful for writing pipeline tests that verify logic, not LLM output.
    """
    return mock_response
