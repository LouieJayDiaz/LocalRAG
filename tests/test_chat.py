import json
from pathlib import Path
from typing import Any

import pytest

from localrag.chat import LocalChat, ModelError, build_prompt
from localrag.ingestion import Document
from localrag.search import LocalIndex, SearchResult


def test_build_prompt_numbers_sources() -> None:
    prompt = build_prompt(
        "What is the answer?",
        (
            SearchResult(
                document_path=Path("notes.md"),
                chunk_index=0,
                content="The answer is local.",
                score=1.0,
            ),
        ),
    )

    assert "[1] notes.md (chunk 0)" in prompt
    assert "The answer is local." in prompt
    assert "What is the answer?" in prompt


def test_local_chat_sends_grounded_prompt_and_returns_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LocalIndex(tmp_path / "index.sqlite3") as index:
        index.add_documents(
            [Document(tmp_path / "notes.md", "The answer is stored locally.")],
            max_characters=100,
        )
        captured: dict[str, Any] = {}

        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def __iter__(self) -> Any:
                return iter([b'{"response":"It is local. [1]","done":true}\n'])

        def fake_urlopen(request: Any, timeout: float) -> Response:
            captured["payload"] = json.loads(request.data)
            captured["timeout"] = timeout
            return Response()

        monkeypatch.setattr("localrag.chat.urlopen", fake_urlopen)
        response = LocalChat(index, model="llama3.2").ask("Where is the answer?")

    assert response.answer == "It is local. [1]"
    assert response.sources[0].document_path.name == "notes.md"
    assert captured["payload"]["model"] == "llama3.2"
    assert captured["payload"]["stream"] is True
    assert "[1]" in captured["payload"]["prompt"]


def test_local_chat_wraps_connection_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LocalIndex(tmp_path / "index.sqlite3") as index:
        monkeypatch.setattr(
            "localrag.chat.urlopen",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("unexpected")
            ),
        )
        with pytest.raises(ModelError, match="Could not connect"):
            LocalChat(index, model="llama3.2").ask("Question")


def test_local_chat_reports_invalid_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with LocalIndex(tmp_path / "index.sqlite3") as index:
        class Response:
            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_: object) -> None:
                pass

            def __iter__(self) -> Any:
                return iter([b"not-json\n"])

        monkeypatch.setattr("localrag.chat.urlopen", lambda *_args, **_kwargs: Response())
        with pytest.raises(ModelError, match="invalid JSON"):
            LocalChat(index, model="llama3.2").ask("Question")


def test_local_chat_rejects_empty_model(tmp_path: Path) -> None:
    with LocalIndex(tmp_path / "index.sqlite3") as index:
        with pytest.raises(ValueError):
            LocalChat(index, model=" ")
