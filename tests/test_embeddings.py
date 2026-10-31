import json

import pytest

from localrag.embeddings import EmbeddingError, NomicOllamaEmbeddings


class Response:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = json.dumps(body).encode()

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *_: object) -> None:
        pass

    def read(self) -> bytes:
        return self.body


@pytest.mark.parametrize(
    "body",
    [
        {"embeddings": [[1.0], ["not a number"]]},
        {"embeddings": [[1.0], [1.0, 2.0]]},
        {"embeddings": [[float("inf")], [1.0]]},
    ],
)
def test_embeddings_reject_invalid_vectors(
    monkeypatch: pytest.MonkeyPatch, body: dict[str, object]
) -> None:
    monkeypatch.setattr(
        "localrag.embeddings.urlopen",
        lambda *_args, **_kwargs: Response(body),
    )

    with pytest.raises(EmbeddingError, match="invalid vector"):
        NomicOllamaEmbeddings()(["one", "two"])


def test_embeddings_returns_normalized_numeric_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "localrag.embeddings.urlopen",
        lambda *_args, **_kwargs: Response({"embeddings": [[1, 2], [3.0, 4.0]]}),
    )

    result = NomicOllamaEmbeddings()(["one", "two"])

    assert result == [[1.0, 2.0], [3.0, 4.0]]
