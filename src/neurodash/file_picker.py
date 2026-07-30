"""
file_picker.py — native OS file dialog for neurodash.

Why this exists:
    Dash runs in a browser. Browsers cannot access the filesystem for
    security reasons, so a standard file input only provides bytes —
    unacceptable for large data files (pl2, avi, xlsx can be hundreds of MB).

    The fix is to get the path directly via a native OS file dialog (QFileDialog).
    But we can't call QFileDialog in-process while Dash is running, so we spawn
    a subprocess that owns its own QApplication, opens the dialog, and prints
    the selected path to stdout. The parent process reads it back.
    On Windows, SetForegroundWindow forces the dialog in front of the browser.

Usage from Dash callback:
    from neurodash.file_picker import pick_file, pick_save_path
    path = pick_file("Select .pl2 file", "Plexon (*.pl2)")
    out = pick_save_path("Export analysis CSV", "CSV (*.csv)", d, "FC33-4_analysis.csv")
"""

import subprocess
import sys


def pick_file(title, filter_str, start_dir=""):
    """Open a native file picker dialog and return the selected path.

    Runs QFileDialog in a subprocess to avoid QApplication conflicts.
    Returns empty string if user cancels.

    Parameters
    ----------
    title : str
    filter_str : str — e.g. "Plexon (*.pl2)"
    start_dir : str — directory to open the dialog in (default: system default)

    Returns
    -------
    str — selected file path, or "" if cancelled
    """
    return _run_dialog("open", title, filter_str, start_dir, "")


def pick_save_path(title, filter_str, start_dir="", default_name=""):
    """Open a native Save-As dialog and return the chosen path.

    Same subprocess trick as pick_file. The dialog is prefilled with
    `default_name` so the common case is one Enter press, and the caller's
    `start_dir` is only a starting point — the user can save anywhere.

    Parameters
    ----------
    title : str
    filter_str : str — e.g. "CSV (*.csv)"
    start_dir : str — directory to open the dialog in
    default_name : str — filename to prefill

    Returns
    -------
    str — chosen file path, or "" if cancelled
    """
    return _run_dialog("save", title, filter_str, start_dir, default_name)


def _run_dialog(mode, title, filter_str, start_dir, default_name):
    result = subprocess.run(
        [sys.executable, __file__, mode, title, filter_str, start_dir, default_name],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    # Called as subprocess — open dialog and print path to stdout
    import sys as _sys
    def _arg(i, default=""):
        return _sys.argv[i] if len(_sys.argv) > i else default

    mode = _arg(1, "open")
    title = _arg(2, "Select file")
    filter_str = _arg(3)
    start_dir = _arg(4)
    default_name = _arg(5)

    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6.QtCore import Qt
    app = QApplication([])
    dialog = QFileDialog(None, title, start_dir, filter_str)
    if mode == "save":
        dialog.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
        dialog.setFileMode(QFileDialog.FileMode.AnyFile)
        dialog.setDefaultSuffix("csv")
        if default_name:
            dialog.selectFile(default_name)
    dialog.setWindowFlags(dialog.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)
    dialog.show()
    # Force window to foreground (Windows only)
    if _sys.platform == "win32":
        import ctypes
        ctypes.windll.user32.SetForegroundWindow(int(dialog.winId()))
    if dialog.exec():
        paths = dialog.selectedFiles()
        print(paths[0] if paths else "")
    else:
        print("")
