"""Windows desktop app for LocalRAG."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import math
import sqlite3
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .chat import LocalChat, ModelError
from .ingestion import IngestionError, ingest_directory, read_document
from .ollama import OllamaClient, OllamaError, OllamaModel
from .paths import default_data_directory
from .search import LocalIndex

SUPPORTED_SUFFIXES = {".docx", ".md", ".pdf", ".txt"}


class RoundedPanel(tk.Frame):
    """A panel with rounded corners."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        background: str,
        border: str,
        radius: int = 16,
        padding: int = 8,
        **kwargs: object,
    ) -> None:
        super().__init__(parent, bg=parent.cget("bg"), bd=0, highlightthickness=0, **kwargs)
        self._background = background
        self._border = border
        self._radius = radius
        self._padding = padding
        self._canvas = tk.Canvas(
            self, bg=parent.cget("bg"), bd=0, highlightthickness=0
        )
        self._canvas.pack(fill=tk.BOTH, expand=True)
        self.content = tk.Frame(self._canvas, bg=background, bd=0)
        self._window = self._canvas.create_window(
            padding, padding, window=self.content, anchor="nw"
        )
        self.bind("<Configure>", self._redraw)

    def _redraw(self, event: tk.Event) -> None:
        width = max(event.width, 2)
        height = max(event.height, 2)
        radius = min(self._radius, width // 2, height // 2)
        self._canvas.delete("panel")
        points = _rounded_polygon(width, height, radius)
        self._canvas.create_polygon(
            points, fill=self._border, outline=self._border, tags="panel"
        )
        inset = max(1, self._padding // 2)
        inner_points = _rounded_polygon(width - inset * 2, height - inset * 2, max(1, radius - inset))
        self._canvas.create_polygon(
            [(x + inset, y + inset) for x, y in inner_points],
            fill=self._background,
            outline=self._background,
            tags="panel",
        )
        self._canvas.tag_lower("panel")
        self._canvas.coords(
            self._window, self._padding, self._padding
        )
        self._canvas.itemconfigure(
            self._window,
            width=max(1, width - self._padding * 2),
            height=max(1, height - self._padding * 2),
        )


def _rounded_polygon(width: int, height: int, radius: int) -> list[tuple[int, int]]:
    """Make a rounded rectangle for a Tk canvas."""
    points: list[tuple[int, int]] = []
    for corner_x, corner_y, start in (
        (radius, radius, 180),
        (width - radius, radius, 270),
        (width - radius, height - radius, 0),
        (radius, height - radius, 90),
    ):
        for step in range(0, 91, 15):
            angle = (start + step) * math.pi / 180
            points.append(
                (
                    round(corner_x + radius * math.cos(angle)),
                    round(corner_y + radius * math.sin(angle)),
                )
            )
    return points


def default_data_path() -> Path:
    """Return the user's local index path."""
    return default_data_directory() / "index.sqlite3"


class LocalRAGApp:
    """The Tkinter app."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("LocalRAG")
        self.root.geometry("1280x820")
        self.root.minsize(1000, 650)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.busy = False
        self.current_chat: LocalChat | None = None
        self.current_sources: dict[str, object] = {}
        self.k = tk.IntVar(value=5)
        self.chunk_size = tk.IntVar(value=1000)
        self.chunk_overlap = tk.IntVar(value=100)
        self.similarity_threshold = tk.DoubleVar(value=0.0)
        self.ollama = OllamaClient()
        self.model_names: list[str] = []
        self.indexed_paths: list[Path] = []
        self.action_buttons: list[ttk.Button] = []

        self.model = tk.StringVar(value="llama3.2")
        self.status = tk.StringVar(value="Ready")
        self.document_count = tk.StringVar(value="0 files")
        self._build_ui()
        self._refresh_document_list()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        self._configure_theme()
        self.root.configure(bg="#141414")

        shell = tk.Frame(self.root, bg="#141414")
        shell.pack(fill=tk.BOTH, expand=True)

        activity = tk.Frame(shell, bg="#171717", width=54)
        activity.pack(side=tk.LEFT, fill=tk.Y)
        activity.pack_propagate(False)
        self._activity_icon(activity, "⌂", True, self.show_home)
        self._activity_icon(activity, "◈", False, self.open_model_manager)
        self._activity_icon(activity, "⚙", False, self.open_settings)

        sidebar = tk.Frame(shell, bg="#202020", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        tk.Label(
            sidebar, text="LocalRAG", bg="#202020", fg="#f0f1f5",
            font=("Segoe UI", 13, "bold"), padx=18, pady=18, anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            sidebar, text="KNOWLEDGE BASE", bg="#202020", fg="#858695",
            font=("Segoe UI", 8, "bold"), padx=18, pady=8, anchor="w",
        ).pack(fill=tk.X, pady=(0, 2))

        self.add_files_button = self._sidebar_button(sidebar, "＋  Add files", self.add_files)
        self.add_folder_button = self._sidebar_button(sidebar, "＋  Add folder", self.add_folder)
        tk.Label(
            sidebar, text="CONVERSATION", bg="#202020", fg="#858695",
            font=("Segoe UI", 8, "bold"), padx=18, pady=14, anchor="w",
        ).pack(fill=tk.X, pady=(12, 2))
        self.new_chat_button = self._sidebar_button(sidebar, "＋  New chat", self.new_chat)
        self.action_buttons.extend(
            [self.add_files_button, self.add_folder_button, self.new_chat_button]
        )
        ttk.Separator(sidebar).pack(fill=tk.X, padx=14, pady=12)
        self.files_label = tk.Label(
            sidebar, textvariable=self.document_count, bg="#202020",
            fg="#858695", font=("Segoe UI", 9), padx=18, anchor="w",
        )
        self.files_label.pack(fill=tk.X, pady=(0, 4))
        document_list_frame = tk.Frame(sidebar, bg="#202020")
        document_list_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 12))
        self.documents = tk.Listbox(
            document_list_frame, height=10, bg="#202020", fg="#c8cad4",
            selectbackground="#34313f", selectforeground="#ffffff",
            highlightthickness=0, borderwidth=0, activestyle="none",
            font=("Segoe UI", 10), relief=tk.FLAT,
        )
        self.documents.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        document_scroll = ttk.Scrollbar(
            document_list_frame, orient=tk.VERTICAL, command=self.documents.yview
        )
        document_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.documents.configure(yscrollcommand=document_scroll.set)
        self.documents.bind("<<ListboxSelect>>", self._show_selected_path)
        self.path_hint = tk.Label(
            sidebar, text="", bg="#202020", fg="#707482", anchor="w",
            justify="left", wraplength=270, padx=22, pady=4,
            font=("Segoe UI", 8),
        )
        self.path_hint.pack(fill=tk.X)
        sidebar_footer = tk.Frame(sidebar, bg="#202020")
        sidebar_footer.pack(fill=tk.X, side=tk.BOTTOM, padx=0, pady=(0, 10))
        ttk.Separator(sidebar_footer).pack(fill=tk.X, padx=14, pady=(0, 8))
        self.clear_button = self._sidebar_button(sidebar_footer, "×  Clear index", self.clear_index)
        self.settings_button = self._sidebar_button(
            sidebar_footer, "⚙  Retrieval settings", self.open_settings
        )
        self.action_buttons.extend([self.clear_button, self.settings_button])

        main = tk.Frame(shell, bg="#141414")
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        header = tk.Frame(main, bg="#1a1a1a", height=58)
        header.pack(fill=tk.X)
        header.pack_propagate(False)
        tk.Label(
            header, text="  LocalRAG  /  New conversation", bg="#1a1a1a",
            fg="#f0f1f5", font=("Segoe UI", 11, "bold"), anchor="w",
        ).pack(side=tk.LEFT, fill=tk.Y)
        self.refresh_models_button = ttk.Button(
            header, text="↻", width=3, command=self.refresh_models
        )
        self.refresh_models_button.pack(side=tk.RIGHT, padx=(6, 12), pady=7)
        self.manage_models_button = ttk.Button(
            header, text="Get models", command=self.open_model_manager
        )
        self.manage_models_button.pack(side=tk.RIGHT, padx=(0, 8), pady=7)
        self.model_combo = ttk.Combobox(
            header, textvariable=self.model, width=20, state="normal"
        )
        self.model_combo.pack(side=tk.RIGHT, pady=8)
        ttk.Label(header, text="Model:").pack(side=tk.RIGHT, padx=(0, 6))
        self.action_buttons.extend(
            [self.model_combo, self.refresh_models_button, self.manage_models_button]
        )

        content = tk.Frame(main, bg="#141414")
        content.pack(fill=tk.BOTH, expand=True, padx=42, pady=28)
        hero = tk.Frame(content, bg="#141414")
        hero.pack(fill=tk.X, pady=(0, 12))
        self.hero_title = tk.Label(
            hero, text="What can I help you find?", bg="#141414", fg="#f2f2f5",
            font=("Segoe UI", 18, "bold"), anchor="center",
        )
        self.hero_title.pack(fill=tk.X)
        self.hero_subtitle = tk.Label(
            hero, text="Private answers, grounded in the documents you choose.",
            bg="#141414", fg="#858695", font=("Segoe UI", 9), anchor="center",
        )
        self.hero_subtitle.pack(fill=tk.X, pady=(4, 0))

        output_frame = RoundedPanel(
            content, background="#1b1c1f", border="#292a2e", radius=14, padding=4
        )
        output_frame.pack(fill=tk.BOTH, expand=True)
        self.output = tk.Text(
            output_frame.content, wrap=tk.WORD, state=tk.DISABLED, bg="#1b1c1f",
            fg="#d4d4d4", insertbackground="#ffffff", selectbackground="#264f78",
            relief=tk.FLAT, borderwidth=0, padx=28, pady=22,
            font=("Segoe UI", 11), spacing3=8,
        )
        self.output.tag_configure(
            "welcome", foreground="#aeb4c0", font=("Segoe UI", 11), justify="center",
            lmargin1=80, lmargin2=80, rmargin=80, spacing1=5
        )
        self.output.tag_configure(
            "user_label", foreground="#b7a9ff", font=("Segoe UI", 9, "bold"),
            justify="right", lmargin1=150, lmargin2=150, rmargin=8, spacing1=12,
        )
        self.output.tag_configure(
            "user_message", foreground="#f3f0ff", background="#39334f",
            justify="right", lmargin1=150, lmargin2=150, rmargin=8,
            spacing1=8, spacing3=12,
        )
        self.output.tag_configure(
            "assistant_label", foreground="#8ce2c3", font=("Segoe UI", 9, "bold"),
            lmargin1=8, spacing1=16,
        )
        self.output.tag_configure(
            "assistant_message", foreground="#e8e8eb", background="#2a2b2f",
            lmargin1=8, rmargin=150, spacing1=8, spacing3=12,
        )
        self.output.tag_configure(
            "sources_header", foreground="#858c98", font=("Segoe UI", 9, "bold"),
            lmargin1=4, spacing1=10,
        )
        self.output.tag_configure("citation", foreground="#4daafc", underline=True)
        self.output.tag_bind("citation", "<Button-1>", self.open_source)
        self.output.tag_bind("citation", "<Enter>", lambda _event: self.output.configure(cursor="hand2"))
        self.output.tag_bind("citation", "<Leave>", lambda _event: self.output.configure(cursor=""))
        self.output.pack(fill=tk.BOTH, expand=True)
        self._write_output(
            "Welcome to LocalRAG\n\n"
            "Add files from the Explorer, then ask a question below.\n"
            "Your documents stay on this computer.",
            "welcome",
        )

        question_frame = RoundedPanel(
            content, background="#202125", border="#2d2e33", radius=14, padding=5
        )
        question_frame.pack(fill=tk.X, pady=(12, 0))
        self.question = tk.Text(
            question_frame.content, height=2, wrap=tk.WORD, bg="#202125", fg="#e0e1e8",
            insertbackground="#ffffff", relief=tk.FLAT, borderwidth=1,
            highlightbackground="#202125", highlightcolor="#8e7dff",
            padx=14, pady=9, font=("Segoe UI", 11),
        )
        self.question.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
        self.question.bind("<Return>", self._question_return)
        self.stop_button = ttk.Button(question_frame.content, text="Stop  ■", command=self.stop)
        self.stop_button.pack(side=tk.RIGHT, ipadx=10, ipady=4)
        self.stop_button.pack_forget()
        self.ask_button = ttk.Button(question_frame.content, text="Send  ➜", command=self.ask)
        self.ask_button.pack(side=tk.RIGHT, ipadx=10, ipady=4)
        self.action_buttons.append(self.ask_button)

        status_bar = tk.Frame(self.root, bg="#6d5bd0", height=25)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(status_bar, textvariable=self.status, bg="#6d5bd0", fg="#ffffff",
                 font=("Segoe UI", 9), padx=10, anchor="w").pack(side=tk.LEFT)
        tk.Label(status_bar, text="● 100% local", bg="#6d5bd0", fg="#ffffff",
                 font=("Segoe UI", 9), padx=10).pack(side=tk.RIGHT)

    def _configure_theme(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", background="#2b2c31", foreground="#f0f1f5",
                        borderwidth=0, padding=(12, 7), font=("Segoe UI", 9))
        style.map("TButton", background=[("active", "#403c59"), ("disabled", "#25262a")])
        style.configure("TLabel", background="#1a1a1a", foreground="#c8cad4",
                        font=("Segoe UI", 9))
        style.configure("TEntry", fieldbackground="#242529", foreground="#d4d4d4",
                        insertcolor="#ffffff", borderwidth=1)
        style.configure(
            "TCombobox", fieldbackground="#242529", background="#2b2c31",
            foreground="#e0e1e8", arrowcolor="#b7a9ff", borderwidth=0,
        )

    def _activity_icon(
        self, parent: tk.Frame, text: str, active: bool, command: object
    ) -> None:
        icon = tk.Label(
            parent, text=text, bg="#403c59" if active else "#171920",
            fg="#ffffff" if active else "#858695", font=("Segoe UI Symbol", 17),
            width=3, pady=10,
        )
        icon.pack(pady=(8, 0))
        icon.bind("<Button-1>", lambda _event: command())
        icon.bind("<Enter>", lambda _event: icon.configure(fg="#ffffff"))
        icon.bind(
            "<Leave>",
            lambda _event: icon.configure(fg="#ffffff" if active else "#858695"),
        )

    def show_home(self) -> None:
        self.question.focus_set()
        self.status.set("Ready")

    def _sidebar_button(self, parent: tk.Frame, text: str, command: object) -> ttk.Button:
        button = ttk.Button(parent, text=text, command=command)
        button.pack(fill=tk.X, padx=12, pady=3)
        return button

    def _write_output(self, text: str, tag: str | None = None) -> None:
        self.output.configure(state=tk.NORMAL)
        self.output.delete("1.0", tk.END)
        if tag is None:
            self.output.insert(tk.END, text)
        else:
            self.output.insert(tk.END, text, tag)
        self.output.configure(state=tk.DISABLED)

    def _compact_hero(self) -> None:
        self.hero_title.configure(
            text="Conversation", font=("Segoe UI", 14, "bold"), fg="#d5d9e0"
        )
        self.hero_subtitle.configure(
            text="Answers grounded in your local documents.", fg="#707782",
            font=("Segoe UI", 9),
        )

    def _question_return(self, event: tk.Event) -> str:
        if event.state & 0x0001:
            return
        self.ask()
        return "break"

    def add_files(self) -> None:
        paths = filedialog.askopenfilenames(
            title="Choose documents",
            filetypes=[
                ("Supported documents", "*.txt *.md *.pdf *.docx"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self._index_paths([Path(path) for path in paths])

    def add_folder(self) -> None:
        path = filedialog.askdirectory(title="Choose a document folder")
        if path:
            self._index_folder(Path(path))

    def _index_paths(self, paths: list[Path]) -> None:
        self._set_busy("Reading and indexing documents...")
        self.executor.submit(self._index_files, paths)

    def _index_folder(self, directory: Path) -> None:
        self._set_busy("Reading and indexing folder...")
        self.executor.submit(self._index_directory, directory)

    def _index_files(self, paths: list[Path]) -> None:
        try:
            documents = [read_document(path) for path in paths]
            with LocalIndex(default_data_path()) as index:
                count = index.add_documents(
                    documents,
                    max_characters=self.chunk_size.get(),
                    overlap_characters=self.chunk_overlap.get(),
                )
        except (IngestionError, OSError, ValueError) as error:
            self.root.after(0, lambda: self._show_error(str(error)))
            return
        self.root.after(0, lambda: self._indexed(count, len(documents)))

    def _index_directory(self, directory: Path) -> None:
        try:
            documents = ingest_directory(directory)
            with LocalIndex(default_data_path()) as index:
                count = index.add_documents(
                    documents,
                    max_characters=self.chunk_size.get(),
                    overlap_characters=self.chunk_overlap.get(),
                )
        except (IngestionError, OSError, ValueError) as error:
            self.root.after(0, lambda: self._show_error(str(error)))
            return
        self.root.after(0, lambda: self._indexed(count, len(documents)))

    def _indexed(self, chunk_count: int, document_count: int) -> None:
        self.busy = False
        self._set_controls_enabled(True)
        self._refresh_document_list()
        self.refresh_models()
        self.document_count.set(
            f"{document_count} file(s) • {chunk_count} chunk(s) added"
        )
        self.status.set("Ready. Ask a question about your documents.")

    def _refresh_document_list(self) -> None:
        try:
            with LocalIndex(default_data_path()) as index:
                paths = index.list_documents()
        except (OSError, ValueError) as error:
            self._show_error(str(error))
            return
        self.indexed_paths = paths
        self.documents.delete(0, tk.END)
        for path in paths:
            self.documents.insert(tk.END, self._display_path(path))
        self.document_count.set(f"{len(paths)} file(s)" if paths else "No files")

    def _show_selected_path(self, _event: tk.Event) -> None:
        selection = self.documents.curselection()
        if selection and selection[0] < len(self.indexed_paths):
            self.path_hint.configure(text=str(self.indexed_paths[selection[0]]))

    def _display_path(self, path: Path) -> str:
        text = str(path)
        return text if len(text) <= 48 else f"...{text[-45:]}"

    def refresh_models(self) -> None:
        self.status.set("Checking local models...")
        self.executor.submit(self._load_models)

    def _load_models(self) -> None:
        try:
            models = self.ollama.list_models()
        except OllamaError:
            self.root.after(0, lambda: self.status.set("Ollama is offline"))
            return
        names = [model.name for model in models]
        self.root.after(0, lambda: self._set_models(names))

    def _set_models(self, names: list[str]) -> None:
        self.model_names = names
        self.model_combo["values"] = names
        if names and self.model.get() not in names:
            self.model.set(names[0])
        self.status.set(f"{len(names)} local model(s) available")

    def open_model_manager(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Local models")
        dialog.geometry("560x420")
        dialog.configure(bg="#252526")
        tk.Label(
            dialog, text="Install a local model", bg="#252526", fg="#f0f0f0",
            font=("Segoe UI", 16, "bold"), padx=18, pady=14, anchor="w",
        ).pack(fill=tk.X)
        tk.Label(
            dialog,
            text="Models run locally through Ollama. Downloads are always explicit.",
            bg="#252526", fg="#9da3ad", padx=18, anchor="w",
        ).pack(fill=tk.X)
        recommended = tk.Listbox(
            dialog, bg="#1e1e1e", fg="#d4d4d4", highlightthickness=0,
            relief=tk.FLAT, font=("Segoe UI", 10), height=6,
        )
        for name in ("llama3.2", "qwen2.5:7b", "mistral"):
            recommended.insert(tk.END, name)
        recommended.pack(fill=tk.X, padx=18, pady=16)
        progress = ttk.Progressbar(dialog, mode="indeterminate")
        progress.pack(fill=tk.X, padx=18, pady=(0, 8))
        status = tk.StringVar(value="Choose a model to download.")
        tk.Label(dialog, textvariable=status, bg="#252526", fg="#9da3ad",
                 padx=18, anchor="w").pack(fill=tk.X)

        def download() -> None:
            selection = recommended.curselection()
            if not selection:
                messagebox.showinfo("Models", "Choose a model first.", parent=dialog)
                return
            name = recommended.get(selection[0])
            progress.start(12)
            status.set(f"Downloading {name}...")
            self.executor.submit(self._download_model, name, dialog, status, progress)

        ttk.Button(dialog, text="Download selected model", command=download).pack(
            anchor="e", padx=18, pady=16
        )

    def _download_model(
        self, name: str, dialog: tk.Toplevel, status: tk.StringVar,
        progress: ttk.Progressbar,
    ) -> None:
        try:
            self.ollama.pull_model(
                name, progress=lambda event: self.root.after(
                    0, lambda: status.set(str(event.get("status", "Downloading...")))
                )
            )
        except OllamaError as error:
            self.root.after(0, lambda: self._show_error(str(error)))
            return
        self.root.after(0, lambda: (progress.stop(), status.set("Installed."), self.refresh_models()))

    def clear_index(self) -> None:
        if self.busy:
            return
        if not messagebox.askyesno(
            "Clear index", "Remove all indexed documents and chat history?"
        ):
            return
        try:
            with LocalIndex(default_data_path()) as index:
                index.clear()
        except (OSError, ValueError) as error:
            self._show_error(str(error))
            return
        self.documents.delete(0, tk.END)
        self.document_count.set("No files")
        self._write_output(
            "Index cleared\n\nAdd files from the Explorer to start a new chat."
        )
        self.status.set("Index cleared.")

    def ask(self) -> None:
        question = self.question.get("1.0", tk.END).strip()
        model = self.model.get().strip()
        if not question:
            messagebox.showinfo("LocalRAG", "Type a question first.")
            return
        if not model:
            messagebox.showinfo("LocalRAG", "Enter a local model name first.")
            return
        self._set_busy("Asking the local model...")
        self.current_chat = LocalChat(
            LocalIndex(default_data_path()), model=model
        )
        self.executor.submit(
            self._ask, question, self.current_chat, self.k.get(),
            self.similarity_threshold.get()
        )

    def _ask(
        self, question: str, chat: LocalChat, limit: int, threshold: float
    ) -> None:
        try:
            response = chat.ask(question, limit=limit, threshold=threshold)
        except (ModelError, OSError, ValueError, RuntimeError, sqlite3.Error) as error:
            chat.index.close()
            if isinstance(error, ModelError) and str(error) == "Generation stopped":
                self.root.after(0, self._show_cancelled)
                return
            self.root.after(0, lambda: self._show_error(str(error)))
            return
        chat.index.close()
        self.root.after(0, lambda: self._show_response(question, response.answer, response.sources))

    def _show_response(self, question: str, answer: str, sources: tuple[object, ...]) -> None:
        self.busy = False
        self._set_controls_enabled(True)
        self.current_chat = None
        self._compact_hero()
        self.output.configure(state=tk.NORMAL)
        self.output.insert(tk.END, "\n\nYou", "user_label")
        self.output.insert(tk.END, f"\n{question}", "user_message")
        self.output.insert(tk.END, "\n\nLocalRAG", "assistant_label")
        self.output.insert(tk.END, f"\n{answer}", "assistant_message")
        if sources:
            self.output.insert(tk.END, "\n\nSources:\n", "sources_header")
            for number, source in enumerate(sources, start=1):
                label = (
                    f"[{number}] {self._display_path(source.document_path)} "
                    f"(chunk {source.chunk_index}, relevance {source.score:.0%})"
                )
                start = self.output.index(tk.END)
                self.output.insert(tk.END, label + "\n")
                self.output.tag_add("citation", start, f"{start} lineend")
                self.current_sources[str(number)] = source
            self.output.insert(tk.END, "\n", "sources_header")
        self.output.configure(state=tk.DISABLED)
        self.question.delete("1.0", tk.END)
        self.status.set("Ready.")

    def _set_busy(self, message: str) -> None:
        self.busy = True
        self._set_controls_enabled(False)
        self.status.set(message)
        self.ask_button.pack_forget()
        self.stop_button.pack(side=tk.RIGHT, ipadx=10, ipady=4)

    def _show_error(self, message: str) -> None:
        self.busy = False
        self._set_controls_enabled(True)
        self.status.set("Ready.")
        self.stop_button.pack_forget()
        self.ask_button.pack(side=tk.RIGHT, ipadx=10, ipady=4)
        messagebox.showerror("LocalRAG", message)

    def _show_cancelled(self) -> None:
        self.busy = False
        self.current_chat = None
        self._set_controls_enabled(True)
        self.stop_button.pack_forget()
        self.ask_button.pack(side=tk.RIGHT, ipadx=10, ipady=4)
        self.status.set("Generation stopped.")

    def _set_controls_enabled(self, enabled: bool) -> None:
        state = tk.NORMAL if enabled else tk.DISABLED
        for button in self.action_buttons:
            button.configure(state=state)

    def stop(self) -> None:
        if self.current_chat is not None:
            self.current_chat.stop()
            self.status.set("Stop requested. Waiting for the local model...")

    def new_chat(self) -> None:
        if self.busy:
            return
        self._write_output("New chat\n\nAdd files from the Explorer, then ask a question.")
        self.current_sources.clear()

    def open_source(self, _event: tk.Event) -> None:
        index = self.output.index(tk.CURRENT)
        tags = self.output.tag_names(index)
        for tag in tags:
            if tag == "citation":
                line = self.output.get(f"{index} linestart", f"{index} lineend")
                number = line.split("]", 1)[0].removeprefix("[")
                source = self.current_sources.get(number)
                if source is not None:
                    self._show_source(source)

    def _show_source(self, source: object) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Source passage")
        dialog.geometry("760x420")
        dialog.configure(bg="#1e1e1e")
        tk.Label(
            dialog,
            text=(
                f"{source.document_path}  •  chunk {source.chunk_index}  •  "
                f"relevance {source.score:.0%}"
            ),
                 bg="#1e1e1e", fg="#4daafc", anchor="w", padx=16, pady=12).pack(fill=tk.X)
        text = tk.Text(dialog, wrap=tk.WORD, bg="#252526", fg="#d4d4d4",
                       relief=tk.FLAT, padx=16, pady=16)
        text.pack(fill=tk.BOTH, expand=True, padx=16, pady=(0, 16))
        text.insert("1.0", source.content)
        text.configure(state=tk.DISABLED)

    def open_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("Retrieval settings")
        dialog.geometry("360x230")
        dialog.configure(bg="#252526")
        fields = [
            ("Chunks to retrieve (k)", self.k, 1, 100),
            ("Similarity threshold", self.similarity_threshold, 0, 1),
            ("Chunk size", self.chunk_size, 100, 10000),
            ("Chunk overlap", self.chunk_overlap, 0, 5000),
        ]
        for row, (label, variable, minimum, maximum) in enumerate(fields):
            tk.Label(dialog, text=label, bg="#252526", fg="#d4d4d4",
                     anchor="w").grid(row=row, column=0, padx=18, pady=12, sticky="w")
            tk.Spinbox(dialog, from_=minimum, to=maximum, increment=0.05 if label == "Similarity threshold" else 1, textvariable=variable,
                       width=10).grid(row=row, column=1, padx=18, pady=12)
        ttk.Button(dialog, text="Save", command=dialog.destroy).grid(
            row=len(fields), column=1, padx=18, pady=10, sticky="e"
        )

    def close(self) -> None:
        self.executor.shutdown(wait=False, cancel_futures=True)
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    LocalRAGApp(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
