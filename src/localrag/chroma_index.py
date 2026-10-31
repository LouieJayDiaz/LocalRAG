"""Optional ChromaDB-backed vector store."""

from pathlib import Path
from typing import Iterable

from .chunking import Chunk, chunk_documents
from .embeddings import NomicOllamaEmbeddings
from .ingestion import Document
from .search import SearchResult


class ChromaIndex:
    """Persist document chunks in ChromaDB with nomic embeddings."""

    def __init__(self, database_path: Path) -> None:
        try:
            import chromadb
        except ImportError as error:
            raise RuntimeError(
                "ChromaDB support requires the rag extra: pip install 'localrag[rag]'"
            ) from error
        self.database_path = database_path.expanduser()
        self.database_path.mkdir(parents=True, exist_ok=True)
        client = chromadb.PersistentClient(path=str(self.database_path))
        self.collection = client.get_or_create_collection(
            name="localrag_documents",
            embedding_function=NomicOllamaEmbeddings(),
        )

    def __enter__(self) -> "ChromaIndex":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def list_documents(self) -> list[Path]:
        data = self.collection.get(include=["metadatas"])
        paths = {
            str(metadata["document_path"])
            for metadata in data.get("metadatas", [])
            if metadata and metadata.get("document_path")
        }
        return sorted((Path(path) for path in paths), key=str)

    def clear(self) -> None:
        ids = self.collection.get()["ids"]
        if ids:
            self.collection.delete(ids=ids)

    def add_documents(
        self,
        documents: Iterable[Document],
        *,
        max_characters: int = 1_000,
        overlap_characters: int | None = None,
    ) -> int:
        document_list = list(documents)
        chunks = chunk_documents(
            document_list,
            max_characters=max_characters,
            overlap_characters=overlap_characters,
        )
        paths = {str(document.path.resolve()) for document in document_list}
        existing = self.collection.get(where={"document_path": {"$in": list(paths)}})
        if not chunks:
            if existing["ids"]:
                self.collection.delete(ids=existing["ids"])
            return 0
        new_ids = [
            f"{chunk.document_path.resolve()}::{chunk.index}" for chunk in chunks
        ]
        self.collection.upsert(
            ids=new_ids,
            documents=[chunk.content for chunk in chunks],
            metadatas=[
                {
                    "document_path": str(chunk.document_path.resolve()),
                    "chunk_index": chunk.index,
                }
                for chunk in chunks
            ],
        )
        stale_ids = [identifier for identifier in existing["ids"] if identifier not in new_ids]
        if stale_ids:
            self.collection.delete(ids=stale_ids)
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
        threshold: float = 0.0,
        semantic_fallback: bool = False,
    ) -> list[SearchResult]:
        if limit <= 0:
            raise ValueError("limit must be greater than zero")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        del semantic_fallback
        result = self.collection.query(query_texts=[query], n_results=limit)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        results: list[SearchResult] = []
        for document, metadata, distance in zip(
            documents, metadatas, distances, strict=True
        ):
            score = 1.0 / (1.0 + float(distance))
            if score >= threshold:
                results.append(
                    SearchResult(
                        document_path=Path(metadata["document_path"]),
                        chunk_index=int(metadata["chunk_index"]),
                        content=document,
                        score=score,
                    )
                )
        return results
