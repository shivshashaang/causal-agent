"""Cached access to the Claude API through the raw Anthropic SDK.

Every response is stored on disk, keyed by a hash of the full request (model,
system prompt, messages, tools and options). Re-running the same request
reads the stored response instead of calling the API, which makes runs
reproducible and free to repeat. The API key is never part of a request, so
it is never written to the cache.
"""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import anthropic

Request = dict[str, Any]
Response = dict[str, Any]

# Fields of each response content block that are sent back to the API as part
# of the conversation. Thinking blocks are returned exactly as received.
_REPLAYED_FIELDS = {
    "text": ("type", "text"),
    "tool_use": ("type", "id", "name", "input"),
    "thinking": ("type", "thinking", "signature"),
    "redacted_thinking": ("type", "data"),
}


class LLMClient(Protocol):
    def create(self, request: Request) -> Response:
        """Send one Messages API request and return the response as a dict."""
        ...


class AnthropicClient:
    """The real client. The key is passed explicitly; no environment lookup."""

    def __init__(self, api_key: str) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)

    def create(self, request: Request) -> Response:
        return self._client.messages.create(**request).to_dict()


def request_key(request: Request) -> str:
    """Stable hash of a request: same prompt and model, same key."""
    canonical = json.dumps(request, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class CachedLLM:
    """Looks a request up on disk first and calls the API only on a miss.

    The client is created lazily, so a fully cached run needs no API key.
    """

    def __init__(
        self, client_factory: Callable[[], LLMClient], cache_dir: Path
    ) -> None:
        self._client_factory = client_factory
        self._client: LLMClient | None = None
        self._cache_dir = Path(cache_dir)
        self.hits = 0
        self.misses = 0

    def create(self, request: Request) -> Response:
        path = self._cache_dir / f"{request_key(request)}.json"
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text(encoding="utf-8"))["response"]

        if self._client is None:
            self._client = self._client_factory()
        response = self._client.create(request)
        self.misses += 1
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        entry = {"request": request, "response": response}
        path.write_text(
            json.dumps(entry, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        return response


def assistant_content(response: Response) -> list[dict[str, Any]]:
    """Turn a response's content blocks into an assistant message's content."""
    blocks = []
    for block in response.get("content", []):
        fields = _REPLAYED_FIELDS.get(block.get("type"))
        if fields is None:
            blocks.append(block)
        else:
            blocks.append({name: block[name] for name in fields if name in block})
    return blocks


def response_text(response: Response) -> str:
    """All text blocks of a response, joined."""
    return "\n".join(
        block["text"]
        for block in response.get("content", [])
        if block.get("type") == "text"
    )


def tool_uses(response: Response) -> list[dict[str, Any]]:
    return [
        block
        for block in response.get("content", [])
        if block.get("type") == "tool_use"
    ]
