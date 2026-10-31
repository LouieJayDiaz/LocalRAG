# LocalRAG

I built LocalRAG because I wanted to ask questions about my own files. It runs
as a website on your computer and uses Ollama or LM Studio to answer questions.
It can read `.txt`, `.md`, `.pdf`, and `.docx` files.

This is still a work in progress, so some parts may be rough.

## Run it

You need Python 3.11 or newer. Install [Ollama](https://ollama.com/download/windows)
or [LM Studio](https://lmstudio.ai/) if you want to chat with a model.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[documents,web]"
py -m localrag.web
```

Open http://127.0.0.1:8000, add some files, and scan for models. LocalRAG shows
models it finds in Ollama or LM Studio. It doesn't download or pick a model for
you. Install or load one in its app, then scan again.

When you chat, LocalRAG sends the question and matching passages from your
files to the model you chose. With Ollama or LM Studio running on your computer,
that stays on your computer. The server starts on `127.0.0.1`, so other devices
can't reach it by default. If you change the host to `0.0.0.0`, devices on your
network may be able to use the site and its API.

LocalRAG saves its index and uploaded copies in the platform's application-data
directory (`%LOCALAPPDATA%\LocalRAG` on Windows, `~/Library/Application
Support/LocalRAG` on macOS, or `$XDG_DATA_HOME/LocalRAG` / `~/.local/share/LocalRAG`
on Linux). Uploads can be up to 25 MB each. **Clear index** removes the index
and those uploaded copies. It doesn't remove the original files from your
computer.

To build the Windows desktop executable, run `build_exe.ps1` in PowerShell
after installing Python 3.11 or newer. The output is `dist\LocalRAG.exe`.

## Search options

By default, LocalRAG uses a basic SQLite search. To use ChromaDB and Ollama's
`nomic-embed-text` model instead, install them and set the backend before
starting the app:

```powershell
py -m pip install -e ".[documents,web,rag]"
ollama pull nomic-embed-text
$env:LOCALRAG_VECTOR_BACKEND = "chroma"
py -m localrag.web
```

The SQLite and ChromaDB indexes are stored separately. If you switch between
them, upload your files again so the selected index has them. The SQLite search
is a simple fallback; ChromaDB is more useful when your question uses different
words from the document.

If you installed LocalRAG in the active virtual environment, you can also run
it with `localrag`.

## Run the tests

```powershell
py -m pip install -e ".[documents,test,web]"
py -m pytest
```
