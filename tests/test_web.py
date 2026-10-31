from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from localrag.web import MAX_UPLOAD_BYTES, LocalRAGServer, create_app


class UnusedOllama:
    def list_models(self, *, timeout_seconds: float) -> list[Any]:
        return []


def client(tmp_path: Path) -> TestClient:
    server = LocalRAGServer(data_directory=tmp_path, ollama=UnusedOllama())
    return TestClient(create_app(server))


def test_uploads_with_same_name_are_kept_separately(tmp_path: Path) -> None:
    with client(tmp_path) as browser:
        response = browser.post(
            "/api/documents",
            files=[
                ("uploads", ("notes.md", b"first document", "text/markdown")),
                ("uploads", ("notes.md", b"second document", "text/markdown")),
            ],
        )

        assert response.status_code == 200
        assert response.json()["documents_added"] == 2
        documents = browser.get("/api/state").json()["documents"]

    assert len(documents) == 2
    assert len(set(documents)) == 2
    assert all(name.startswith("notes-") for name in documents)


def test_clear_removes_indexed_upload_copies(tmp_path: Path) -> None:
    with client(tmp_path) as browser:
        assert browser.post(
            "/api/documents",
            files={"uploads": ("notes.md", b"document", "text/markdown")},
        ).status_code == 200
        assert list((tmp_path / "uploads").iterdir())

        response = browser.post("/api/clear")

        assert response.status_code == 200
        assert browser.get("/api/state").json()["document_count"] == 0

    assert list((tmp_path / "uploads").iterdir()) == []


def test_oversized_upload_is_removed_without_indexing(tmp_path: Path) -> None:
    with client(tmp_path) as browser:
        response = browser.post(
            "/api/documents",
            files={
                "uploads": (
                    "large.md",
                    b"x" * (MAX_UPLOAD_BYTES + 1),
                    "text/markdown",
                )
            },
        )

        assert response.status_code == 400
        assert browser.get("/api/state").json()["document_count"] == 0

    assert list((tmp_path / "uploads").iterdir()) == []
