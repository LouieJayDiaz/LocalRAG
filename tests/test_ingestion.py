from pathlib import Path

import pytest

from localrag.ingestion import IngestionError, ingest_directory, read_document


def test_read_document_returns_utf8_content_and_resolved_path(tmp_path: Path) -> None:
    source = tmp_path / "notes.md"
    source.write_text("# Local notes\n\nPrivate content.", encoding="utf-8")

    document = read_document(source)

    assert document.path == source.resolve()
    assert document.content == "# Local notes\n\nPrivate content."


def test_read_document_accepts_utf8_bom(tmp_path: Path) -> None:
    source = tmp_path / "bom.txt"
    source.write_text("BOM content", encoding="utf-8-sig")

    assert read_document(source).content == "BOM content"


@pytest.mark.parametrize("filename", ["image.png", "no_extension"])
def test_read_document_rejects_unsupported_types(tmp_path: Path, filename: str) -> None:
    source = tmp_path / filename
    source.write_text("content", encoding="utf-8")

    with pytest.raises(IngestionError, match="Unsupported document type"):
        read_document(source)


def test_read_document_rejects_empty_content(tmp_path: Path) -> None:
    source = tmp_path / "empty.txt"
    source.write_text("  \n", encoding="utf-8")

    with pytest.raises(IngestionError, match="Document is empty"):
        read_document(source)


def test_ingest_directory_recurses_in_path_order(tmp_path: Path) -> None:
    (tmp_path / "nested").mkdir()
    (tmp_path / "z.txt").write_text("z", encoding="utf-8")
    (tmp_path / "nested" / "a.md").write_text("a", encoding="utf-8")
    (tmp_path / "ignored.rtf").write_text("ignored", encoding="utf-8")

    documents = ingest_directory(tmp_path)

    assert [document.path.name for document in documents] == ["a.md", "z.txt"]


def test_read_document_reports_missing_pdf_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "notes.pdf"
    source.write_bytes(b"not a real PDF")

    real_import = __import__

    def import_without_pypdf(name: str, *args: object, **kwargs: object) -> object:
        if name == "pypdf":
            raise ImportError("test dependency unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", import_without_pypdf)

    with pytest.raises(IngestionError, match="PDF support requires"):
        read_document(source)
