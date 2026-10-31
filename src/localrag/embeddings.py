"""Embedding providers for document retrieval."""

import json
import math
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class EmbeddingError(RuntimeError):
    """Raised when a local embedding provider cannot create vectors."""


class NomicOllamaEmbeddings:
    """Use Ollama's local nomic-embed-text model for embeddings."""

    def __init__(
        self,
        *,
        endpoint: str = "http://127.0.0.1:11434/api/embed",
        model: str = "nomic-embed-text",
    ) -> None:
        self.endpoint = endpoint
        self.model = model

    def __call__(self, input: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": input}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8"))
            embeddings = body["embeddings"]
        except (
            HTTPError,
            URLError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            KeyError,
            TypeError,
        ) as error:
            raise EmbeddingError(
                "nomic-embed-text is unavailable. Install it in Ollama or use the built-in fallback."
            ) from error
        if (
            not isinstance(embeddings, list)
            or len(embeddings) != len(input)
            or not embeddings
            or not all(isinstance(vector, list) and vector for vector in embeddings)
        ):
            raise EmbeddingError("The embedding provider returned an invalid response.")
        dimension = len(embeddings[0])
        if any(
            len(vector) != dimension
            or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in vector)
            for vector in embeddings
        ):
            raise EmbeddingError("The embedding provider returned invalid vector values.")
        return [[float(value) for value in vector] for vector in embeddings]
