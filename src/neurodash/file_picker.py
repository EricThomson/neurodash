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
    from neurodash.file_picker import pick_file
    path = pick_file("Select .pl2 file", "Plexon (*.pl2)")
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
    result = subprocess.run(
        [sys.executable, __file__, title, filter_str, start_dir],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


if __name__ == "__main__":
    # Called as subprocess — open dialog and print path to stdout
    import sys as _sys
    title = _sys.argv[1] if len(_sys.argv) > 1 else "Select file"
    filter_str = _sys.argv[2] if len(_sys.argv) > 2 else ""
    start_dir = _sys.argv[3] if len(_sys.argv) > 3 else ""

    from PyQt6.QtWidgets import QApplication, QFileDialog
    from PyQt6.QtCore import Qt
    app = QApplication([])
    dialog = QFileDialog(None, title, start_dir, filter_str)
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
