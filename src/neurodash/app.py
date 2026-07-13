"""neurodash Dash app: factory + the `neurodash` CLI entry point."""

import threading
import webbrowser
from pathlib import Path

from dash import Dash

from neurodash.layout import make_layout
import neurodash.callbacks  # noqa: F401 — registers callbacks

# Assets live inside the package so they ship with a pip install and resolve
# regardless of the working directory.
_ASSETS = str(Path(__file__).parent / "assets")

HOST = "127.0.0.1"
PORT = 8050


def create_app():
    """Build and return the neurodash Dash app."""
    app = Dash(__name__, assets_folder=_ASSETS, suppress_callback_exceptions=True)
    app.layout = make_layout()
    return app


def main():
    """`neurodash` entry point: launch the dashboard and open a browser."""
    app = create_app()
    url = f"http://{HOST}:{PORT}"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"neurodash running at {url}  (Ctrl+C to stop)")
    app.run(host=HOST, port=PORT, debug=False)
