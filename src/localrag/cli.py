"""Command-line entry point for checking local documents."""

import argparse
from pathlib import Path

from .ingestion import IngestionError, ingest_directory, read_document


def main() -> int:
    parser = argparse.ArgumentParser(description="Read local documents for LocalRAG")
    parser.add_argument(
        "path",
        type=Path,
        help="A .txt/.md/.pdf/.docx file or a directory of documents",
    )
    arguments = parser.parse_args()

    try:
        if arguments.path.is_dir():
            documents = ingest_directory(arguments.path)
        else:
            documents = [read_document(arguments.path)]
    except IngestionError as error:
        parser.error(str(error))

    for document in documents:
        print(f"{document.path} ({len(document.content)} characters)")

    print(f"Imported {len(documents)} document(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
