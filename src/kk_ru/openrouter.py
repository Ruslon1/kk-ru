from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class OpenRouterError(RuntimeError):
    def __init__(self, message: str, status: int | None = None, body: str | None = None):
        super().__init__(message)
        self.status = status
        self.body = body


@dataclass(frozen=True)
class ModelInfo:
    id: str
    name: str
    prompt_price: float | None
    completion_price: float | None
    context_length: int | None
    supported_parameters: tuple[str, ...] = ()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "ModelInfo":
        pricing = payload.get("pricing") or {}
        supported_parameters = payload.get("supported_parameters") or []
        if not isinstance(supported_parameters, list):
            supported_parameters = []
        return cls(
            id=str(payload["id"]),
            name=str(payload.get("name") or payload["id"]),
            prompt_price=_price(pricing.get("prompt")),
            completion_price=_price(pricing.get("completion")),
            context_length=_integer(payload.get("context_length")),
            supported_parameters=tuple(
                parameter for parameter in supported_parameters if isinstance(parameter, str)
            ),
        )


@dataclass(frozen=True)
class ChatResult:
    text: str
    response_id: str | None
    model: str | None
    provider: str | None
    usage: dict[str, Any]


def _price(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid model price: {value!r}") from error
    if number == -1:
        return None
    if number < 0:
        raise ValueError(f"model price must be non-negative: {value!r}")
    return number


def _integer(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid integer value: {value!r}") from error
    return number


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    if isinstance(content, dict) and isinstance(content.get("text"), str):
        return content["text"]
    return ""


def estimate_input_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


def estimate_cost(
    model: ModelInfo,
    messages: list[dict[str, Any]],
    max_tokens: int,
) -> float | None:
    if model.prompt_price is None or model.completion_price is None:
        return None
    prompt_tokens = estimate_input_tokens(messages)
    return prompt_tokens * model.prompt_price + max_tokens * model.completion_price


def usage_cost(usage: dict[str, Any], model: ModelInfo) -> float | None:
    reported = _number(usage.get("cost"))
    if reported is not None:
        return reported
    prompt_tokens = _integer(usage.get("prompt_tokens"))
    completion_tokens = _integer(usage.get("completion_tokens"))
    if (
        prompt_tokens is None
        or completion_tokens is None
        or model.prompt_price is None
        or model.completion_price is None
    ):
        return None
    return prompt_tokens * model.prompt_price + completion_tokens * model.completion_price


class OpenRouterClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 120.0,
        opener: Callable[..., Any] | None = None,
    ):
        self.api_key = api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._opener = opener or urlopen

    def _request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
        authenticated: bool = False,
    ) -> dict[str, Any]:
        if authenticated and not self.api_key:
            raise OpenRouterError("OPENROUTER_API_KEY is required for chat completions")

        headers = {"Accept": "application/json"}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = Request(
            f"{self.base_url}/{path.lstrip('/')}",
            data=data,
            headers=headers,
            method=method,
        )
        try:
            with self._opener(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except HTTPError as error:
            body = error.read().decode("utf-8", errors="replace")
            raise OpenRouterError(
                f"OpenRouter HTTP {error.code} for {method} {path}",
                status=error.code,
                body=body,
            ) from error
        except URLError as error:
            raise OpenRouterError(f"OpenRouter request failed for {method} {path}: {error}") from error

        try:
            result = json.loads(body)
        except json.JSONDecodeError as error:
            raise OpenRouterError(f"OpenRouter returned invalid JSON for {method} {path}", body=body) from error
        if not isinstance(result, dict):
            raise OpenRouterError(f"OpenRouter returned a non-object for {method} {path}")
        if result.get("error"):
            error = result["error"]
            message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
            raise OpenRouterError(message, body=body)
        return result

    def list_models(self) -> dict[str, ModelInfo]:
        result = self._request("GET", "models")
        items = result.get("data")
        if not isinstance(items, list):
            raise OpenRouterError("OpenRouter models response has no data list")
        return {item["id"]: ModelInfo.from_payload(item) for item in items if isinstance(item, dict)}

    def complete(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 256,
        reasoning: dict[str, Any] | None = None,
    ) -> ChatResult:
        if not model:
            raise ValueError("model must not be empty")
        if not messages:
            raise ValueError("messages must not be empty")
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if reasoning is not None:
            payload["reasoning"] = reasoning
        result = self._request("POST", "chat/completions", payload, authenticated=True)
        choices = result.get("choices")
        if not isinstance(choices, list) or not choices:
            raise OpenRouterError("OpenRouter response has no choices")
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        content = _content_text(message.get("content") if isinstance(message, dict) else None).strip()
        if not content:
            raise OpenRouterError("OpenRouter response has empty message content")
        usage = result.get("usage")
        return ChatResult(
            text=content,
            response_id=result.get("id"),
            model=result.get("model"),
            provider=result.get("provider"),
            usage=usage if isinstance(usage, dict) else {},
        )
