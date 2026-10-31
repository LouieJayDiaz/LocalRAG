from pathlib import Path

from localrag import paths


def test_default_data_directory_uses_xdg_data_home(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths.sys, "platform", "linux")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))

    assert paths.default_data_directory() == tmp_path / "data" / "LocalRAG"


def test_default_data_directory_uses_localappdata_on_windows(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(paths.sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "appdata"))

    assert paths.default_data_directory() == tmp_path / "appdata" / "LocalRAG"


def test_desktop_module_imports_with_tk_event_annotations() -> None:
    from localrag.desktop import default_data_path

    assert default_data_path().name == "index.sqlite3"
