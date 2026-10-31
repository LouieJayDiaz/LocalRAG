from pathlib import Path

import pytest

from localrag.chunking import Chunk, chunk_document, chunk_documents
from localrag.ingestion import Document


def make_document(content: str, name: str = "notes.md") -> Document:
    return Document(path=Path(name).resolve(), content=content)


def test_chunk_document_respects_limit_and_keeps_overlap() -> None:
    document = make_document("one two three four five six seven eight nine ten")

    chunks = chunk_document(document, max_characters=20, overlap_characters=5)

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert all(len(chunk.content) <= 20 for chunk in chunks)
    assert chunks[0].content.endswith("four")
    assert "four" in chunks[1].content
    assert all(chunk.document_path == document.path for chunk in chunks)


def test_chunk_document_keeps_short_documents_intact() -> None:
    document = make_document("short document")

    assert chunk_document(document, max_characters=100) == [
        Chunk(document_path=document.path, index=0, content="short document")
    ]


@pytest.mark.parametrize(
    ("max_characters", "overlap_characters"),
    [(0, 0), (-1, 0), (10, -1), (10, 10), (10, 11)],
)
def test_chunk_document_rejects_invalid_configuration(
    max_characters: int, overlap_characters: int
) -> None:
    with pytest.raises(ValueError):
        chunk_document(
            make_document("content"),
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )


def test_chunk_documents_preserves_document_order() -> None:
    first = make_document("first", "first.md")
    second = make_document("second", "second.md")

    chunks = chunk_documents([first, second], max_characters=20)

    assert [chunk.document_path.name for chunk in chunks] == ["first.md", "second.md"]
