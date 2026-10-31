"""Small client for Ollama's local API."""

import json
import threading
from dataclasses import dataclass
from typing import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaError(RuntimeError):
    """Raised when Ollama cannot complete a request."""


@dataclass(frozen=True, slots=True)
class OllamaModel:
    """A model installed in Ollama."""

    name: str
    size_bytes: int = 0

    @property
    def size_label(self) -> str:
        if self.size_bytes <= 0:
            return ""
        size = float(self.size_bytes)
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size < 1024 or unit == "TB":
                return f"{size:.1f} {unit}"
            size /= 1024
        return ""


class OllamaClient:
    """Talk to Ollama over HTTP."""

    def __init__(self, endpoint: str = "http://127.0.0.1:11434") -> None:
        self.endpoint = endpoint.rstrip("/")

    def list_models(self, *, timeout_seconds: float = 2.0) -> list[OllamaModel]:
        body = self._request("GET", "/api/tags", timeout_seconds=timeout_seconds)
        models = body.get("models", [])
        if not isinstance(models, list):
            raise OllamaError("Ollama returned an invalid model list")
        return [
            OllamaModel(
                name=str(item.get("name", "")),
                size_bytes=int(item.get("size", 0) or 0),
            )
            for item in models
            if isinstance(item, dict) and item.get("name")
        ]

    def pull_model(
        self,
        name: str,
        *,
        progress: Callable[[dict[str, object]], None] | None = None,
        stop_event: threading.Event | None = None,
    ) -> None:
        """Download a model and report progress."""
        if not name.strip():
            raise ValueError("model name cannot be empty")
        payload = json.dumps({"name": name.strip(), "stream": True}).encode("utf-8")
        request = Request(
            f"{self.endpoint}/api/pull",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=30) as response:
                for raw_line in response:
                    if stop_event is not None and stop_event.is_set():
                        return
                    if not raw_line.strip():
                        continue
                    message = json.loads(raw_line.decode("utf-8"))
                    if progress is not None:
                        progress(message)
                    if message.get("error"):
                        raise OllamaError(str(message["error"]))
        except HTTPError as error:
            raise OllamaError(
                f"Ollama model download failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (URLError, OSError) as error:
            raise OllamaError(f"Could not connect to Ollama: {error}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaError("Ollama returned invalid download progress") from error

    def _request(
        self, method: str, path: str, *, timeout_seconds: float = 10.0
    ) -> dict[str, object]:
        request = Request(f"{self.endpoint}{path}", method=method)
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            raise OllamaError(
                f"Ollama request failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (URLError, OSError) as error:
            raise OllamaError(f"Could not connect to Ollama: {error}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise OllamaError("Ollama returned invalid JSON") from error
        if not isinstance(body, dict):
            raise OllamaError("Ollama returned an invalid response")
        return body
