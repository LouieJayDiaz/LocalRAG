"""Platform-specific paths used by LocalRAG."""

import os
import sys
from pathlib import Path


def default_data_directory() -> Path:
    """Return the directory used for persistent LocalRAG data."""
    if sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    elif sys.platform == "darwin":
        root = Path.home() / "Library" / "Application Support"
    else:
        root = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return root / "LocalRAG"
