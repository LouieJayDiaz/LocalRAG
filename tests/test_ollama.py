import io
import json
from typing import Any

import pytest

from localrag.ollama import OllamaClient, OllamaError


class Response:
    def __init__(self, body: bytes) -> None:
        self.body = io.BytesIO(body)

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def read(self) -> bytes:
        return self.body.read()

    def __iter__(self) -> Any:
        return iter(self.body.readline, b"")


def test_list_models_parses_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "localrag.ollama.urlopen",
        lambda *_args, **_kwargs: Response(
            json.dumps({"models": [{"name": "llama3.2", "size": 1024}]}).encode()
        ),
    )

    models = OllamaClient().list_models()

    assert models[0].name == "llama3.2"
    assert models[0].size_label == "1.0 KB"


def test_pull_model_reports_streamed_progress(monkeypatch: pytest.MonkeyPatch) -> None:
    events: list[dict[str, object]] = []
    body = b'{"status":"downloading","completed":5,"total":10}\n{"status":"success"}\n'
    monkeypatch.setattr("localrag.ollama.urlopen", lambda *_args, **_kwargs: Response(body))

    OllamaClient().pull_model("llama3.2", progress=events.append)

    assert events[0]["status"] == "downloading"
    assert events[-1]["status"] == "success"


def test_pull_model_rejects_empty_name() -> None:
    with pytest.raises(ValueError):
        OllamaClient().pull_model(" ")


def test_client_surfaces_service_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "localrag.ollama.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("offline")),
    )

    with pytest.raises(OllamaError, match="Could not connect"):
        OllamaClient().list_models()
