"""SQLite index and local search."""

import hashlib
import json
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .chunking import Chunk, chunk_documents
from .ingestion import Document

_TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A matching chunk and its score."""

    document_path: Path
    chunk_index: int
    content: str
    score: float


class LocalIndex:
    """Store chunks and hashed embeddings in SQLite."""

    def __init__(self, database_path: Path, *, dimensions: int = 256) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self.database_path = database_path.expanduser()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.dimensions = dimensions
        self._connection = sqlite3.connect(
            self.database_path, check_same_thread=False
        )
        self._connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                document_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL,
                PRIMARY KEY (document_path, chunk_index)
            )
            """
        )
        self._connection.commit()

    def __enter__(self) -> "LocalIndex":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._connection.close()

    def list_documents(self) -> list[Path]:
        """Return indexed document paths."""
        rows = self._connection.execute(
            "SELECT DISTINCT document_path FROM chunks ORDER BY document_path"
        )
        return [Path(row[0]) for row in rows]

    def clear(self) -> None:
        """Remove all indexed chunks."""
        with self._connection:
            self._connection.execute("DELETE FROM chunks")

    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        max_characters: int = 1_000,
        overlap_characters: int | None = None,
    ) -> int:
        """Replace chunks for these documents and return the count."""
        document_list = list(documents)
        chunks = chunk_documents(
            document_list,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
        paths = {str(document.path.resolve()) for document in document_list}
        with self._connection:
            for path in paths:
                self._connection.execute(
                    "DELETE FROM chunks WHERE document_path = ?", (path,)
                )
            self._connection.executemany(
                """
                INSERT INTO chunks(document_path, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk.document_path.resolve()),
                        chunk.index,
                        chunk.content,
                        json.dumps(self.embed(chunk.content)),
                    )
                    for chunk in chunks
                ],
            )
        return len(chunks)

    def add_chunks(self, chunks: Iterable[Chunk]) -> int:
        """Replace explicitly prepared chunks."""
        chunk_list = list(chunks)
        with self._connection:
            for path in {str(chunk.document_path.resolve()) for chunk in chunk_list}:
                self._connection.execute(
                    "DELETE FROM chunks WHERE document_path = ?", (path,)
                )
            self._connection.executemany(
                """
                INSERT INTO chunks(document_path, chunk_index, content, embedding)
                VALUES (?, ?, ?, ?)
                """,
                [
                    (
                        str(chunk.document_path.resolve()),
                        chunk.index,
                        chunk.content,
                        json.dumps(self.embed(chunk.content)),
                    )
                    for chunk in chunk_list
                ],
            )
        return len(chunk_list)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        threshold: float = 0.0,
        semantic_fallback: bool = False,
    ) -> list[SearchResult]:
        """Return matching chunks for a query.

        By default, a chunk must share a token with the query. Chat can opt
        into semantic fallback for broad questions such as "summarize this".
        """
        if not query.strip():
            raise ValueError("query cannot be empty")
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")

        query_tokens = set(_TOKEN_PATTERN.findall(query.casefold()))
        query_embedding = self.embed(query)
        results: list[SearchResult] = []
        for path, index, content, serialized_embedding in self._connection.execute(
            "SELECT document_path, chunk_index, content, embedding FROM chunks"
        ):
            content_tokens = set(_TOKEN_PATTERN.findall(content.casefold()))
            overlap = query_tokens & content_tokens
            if not overlap and not semantic_fallback:
                continue
            lexical_score = len(overlap) / len(query_tokens) if query_tokens else 0.0
            semantic_score = _cosine_similarity(
                query_embedding, json.loads(serialized_embedding)
            )
            score = (lexical_score * 0.7) + (semantic_score * 0.3)
            if score < threshold:
                continue
            results.append(
                SearchResult(Path(path), index, content, score)
            )
        results.sort(key=lambda result: (-result.score, str(result.document_path), result.chunk_index))
        return results[:limit]

    def embed(self, text: str) -> list[float]:
        """Create a local hashed bag-of-words embedding."""
        vector = [0.0] * self.dimensions
        tokens = _TOKEN_PATTERN.findall(text.casefold())
        for token in tokens:
            digest = hashlib.blake2b(
                token.encode("utf-8"), digest_size=8
            ).digest()
            vector[int.from_bytes(digest, "big") % self.dimensions] += 1.0
        magnitude = math.sqrt(sum(value * value for value in vector))
        return [value / magnitude for value in vector] if magnitude else vector


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right, strict=True))
