"""Chat with a local Ollama model using indexed files."""

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .search import LocalIndex, SearchResult


class ModelError(RuntimeError):
    """Raised when the model cannot answer."""


@dataclass(frozen=True, slots=True)
class ChatResponse:
    """An answer and the chunks used for it."""

    answer: str
    sources: tuple[SearchResult, ...]


class LocalChat:
    """Find matching chunks and ask Ollama about them."""

    def __init__(
        self,
        index: LocalIndex,
        *,
        model: str,
        endpoint: str = "http://127.0.0.1:11434/api/generate",
        protocol: str = "ollama",
        timeout_seconds: float = 120.0,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be greater than zero")
        self.index = index
        self.model = model
        self.endpoint = endpoint
        self.protocol = protocol
        self.timeout_seconds = timeout_seconds
        self._stop_event = threading.Event()
        self._response_lock = threading.Lock()
        self._active_response: object | None = None

    def ask(
        self, question: str, *, limit: int = 5, threshold: float = 0.0
    ) -> ChatResponse:
        """Answer a question with numbered citations."""
        if not question.strip():
            raise ValueError("question cannot be empty")
        self._stop_event.clear()

        sources = tuple(
            self.index.search(question, limit=limit, threshold=threshold)
        )
        if not sources:
            sources = tuple(
                self.index.search(
                    question,
                    limit=limit,
                    threshold=threshold,
                    semantic_fallback=True,
                )
            )
        prompt = build_prompt(question, sources)
        answer = self._generate(prompt)
        return ChatResponse(answer=answer, sources=sources)

    def _generate(self, prompt: str) -> str:
        if self._stop_event.is_set():
            raise ModelError("Generation stopped")
        if self.protocol == "openai":
            return self._generate_openai(prompt)
        if self.protocol != "ollama":
            raise ModelError(f"Unsupported local model protocol: {self.protocol}")
        payload = json.dumps(
            {"model": self.model, "prompt": prompt, "stream": True}
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                with self._response_lock:
                    self._active_response = response
                parts: list[str] = []
                for raw_line in response:
                    if self._stop_event.is_set():
                        raise ModelError("Generation stopped")
                    if not raw_line.strip():
                        continue
                    body = json.loads(raw_line.decode("utf-8"))
                    if body.get("error"):
                        raise ModelError(str(body["error"]))
                    part = body.get("response", "")
                    if isinstance(part, str):
                        parts.append(part)
                answer = "".join(parts).strip()
        except HTTPError as error:
            raise ModelError(
                f"Local model request failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (URLError, OSError) as error:
            if self._stop_event.is_set():
                raise ModelError("Generation stopped") from error
            reason = getattr(error, "reason", str(error))
            raise ModelError(f"Could not connect to local model: {reason}") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelError("Local model returned invalid JSON") from error
        finally:
            with self._response_lock:
                self._active_response = None

        if not answer:
            raise ModelError("Local model response did not contain text")
        return answer.strip()

    def _generate_openai(self, prompt: str) -> str:
        payload = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
            }
        ).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                body = json.loads(response.read().decode("utf-8"))
            answer = body["choices"][0]["message"]["content"]
        except HTTPError as error:
            raise ModelError(
                f"Local model request failed with HTTP {error.code}: {error.reason}"
            ) from error
        except (URLError, OSError) as error:
            raise ModelError(f"Could not connect to local model: {error}") from error
        except (KeyError, IndexError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ModelError("LM Studio returned an invalid chat response") from error
        if not isinstance(answer, str) or not answer.strip():
            raise ModelError("Local model response did not contain text")
        return answer.strip()

    def stop(self) -> None:
        """Stop the current generation."""
        self._stop_event.set()
        with self._response_lock:
            response = self._active_response
        if response is not None:
            close = getattr(response, "close", None)
            if callable(close):
                close()


def build_prompt(question: str, sources: tuple[SearchResult, ...]) -> str:
    """Build the prompt sent to the model."""
    if sources:
        context = "\n\n".join(
            f"[{number}] {source.document_path} (chunk {source.chunk_index})\n"
            f"{source.content}"
            for number, source in enumerate(sources, start=1)
        )
    else:
        context = "(No matching document passages were found.)"

    return (
        "You are LocalRAG, a private document assistant. "
        "Answer the user's question using only the document passages below. "
        "If the passages do not contain the answer, say you do not know. "
        "Cite supporting passages with their bracketed source number, such as [1]. "
        "Do not invent citations or facts.\n\n"
        f"DOCUMENT PASSAGES:\n{context}\n\n"
        f"USER QUESTION: {question.strip()}\n\n"
        "ANSWER:"
    )
