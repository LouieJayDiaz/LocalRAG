"""Break documents into chunks."""

from dataclasses import dataclass
from pathlib import Path

from .ingestion import Document


@dataclass(frozen=True, slots=True)
class Chunk:
    """One chunk of a document."""

    document_path: Path
    index: int
    content: str


def chunk_document(
    document: Document,
    *,
    max_characters: int = 1_000,
    overlap_characters: int | None = None,
) -> list[Chunk]:
    """Split a document into overlapping chunks."""
    if max_characters <= 0:
        raise ValueError("max_characters must be greater than zero")
    if overlap_characters is None:
        overlap_characters = min(100, max_characters - 1)
    if overlap_characters < 0:
        raise ValueError("overlap_characters cannot be negative")
    if overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be less than max_characters")

    text = document.content.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    while start < len(text):
        proposed_end = min(start + max_characters, len(text))
        end = proposed_end

        if proposed_end < len(text):
            boundary = text.rfind(" ", start + max_characters // 2, proposed_end)
            if boundary > start:
                end = boundary

        content = text[start:end].strip()
        if content:
            chunks.append(
                Chunk(document_path=document.path, index=len(chunks), content=content)
            )

        if end >= len(text):
            break
        next_start = end - overlap_characters
        start = max(next_start, start + 1)

    return chunks


def chunk_documents(
    documents: list[Document],
    *,
    max_characters: int = 1_000,
    overlap_characters: int | None = None,
) -> list[Chunk]:
    """Chunk documents in the order given."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(
                document,
                max_characters=max_characters,
                overlap_characters=overlap_characters,
            )
        )
    return chunks
