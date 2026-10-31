"""Main LocalRAG building blocks."""

from .chat import ChatResponse, LocalChat, ModelError
from .chunking import Chunk, chunk_document, chunk_documents
from .ingestion import Document, IngestionError, ingest_directory, read_document
from .ollama import OllamaClient, OllamaError, OllamaModel
from .search import LocalIndex, SearchResult

__all__ = [
    "Chunk",
    "ChatResponse",
    "Document",
    "IngestionError",
    "LocalIndex",
    "LocalChat",
    "ModelError",
    "OllamaClient",
    "OllamaError",
    "OllamaModel",
    "SearchResult",
    "chunk_document",
    "chunk_documents",
    "ingest_directory",
    "read_document",
]
