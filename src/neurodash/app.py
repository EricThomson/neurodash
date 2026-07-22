"""neurodash Dash app: factory + the `neurodash` CLI entry point.

The app binds a fixed port, so a second `neurodash` would otherwise fail to bind
while an orphaned server keeps serving stale state (e.g. a process left running
when its terminal was closed rather than Ctrl-C'd). ``main`` guards against that:
it reuses a running instance (just opens the browser) unless ``--restart`` is
given, which stops the old one and starts fresh.
"""

import argparse
import socket
import threading
import time
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
URL = f"http://{HOST}:{PORT}"


def create_app():
    """Build and return the neurodash Dash app."""
    app = Dash(__name__, assets_folder=_ASSETS, suppress_callback_exceptions=True)
    app.layout = make_layout()
    return app


def _server_running():
    """True if something is already accepting connections on the app's port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((HOST, PORT)) == 0


def _stop_running_instance(timeout=5.0):
    """Terminate the neurodash process holding the port (for ``--restart``).

    Best-effort and defensive: uses psutil to map port -> PID, and only kills a
    process whose command line looks like neurodash, so an unrelated app that
    happens to be on the port is left alone. Returns True once the port is free.
    """
    try:
        import psutil
    except ImportError:
        print("psutil isn't installed — can't auto-stop the running instance.")
        return False

    for conn in psutil.net_connections(kind="inet"):
        if not (conn.laddr and conn.laddr.port == PORT
                and conn.status == psutil.CONN_LISTEN and conn.pid):
            continue
        try:
            proc = psutil.Process(conn.pid)
            cmdline = " ".join(proc.cmdline()).lower()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if "neurodash" not in cmdline and "app.py" not in cmdline:
            print(f"Port {PORT} is held by PID {conn.pid} ({proc.name()}), which "
                  f"doesn't look like neurodash — leaving it alone.")
            return False
        print(f"Stopping the running neurodash (PID {conn.pid})...")
        try:
            proc.terminate()
            proc.wait(timeout=timeout)
        except psutil.TimeoutExpired:
            proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    deadline = time.monotonic() + timeout
    while _server_running() and time.monotonic() < deadline:
        time.sleep(0.1)
    return not _server_running()


def _run():
    """Start the server and open a browser once it's up."""
    timer = threading.Timer(1.0, lambda: webbrowser.open(URL))
    timer.daemon = True
    timer.start()
    print(f"neurodash running at {URL}  (Ctrl+C to stop)")
    create_app().run(host=HOST, port=PORT, debug=False)


def main():
    """`neurodash` entry point: launch the dashboard and open a browser.

    Reuses an already-running instance instead of starting a duplicate; pass
    ``--restart`` to stop the running one and start fresh.
    """
    parser = argparse.ArgumentParser(
        prog="neurodash", description="Neurobehavioral data explorer.")
    parser.add_argument(
        "--restart", action="store_true",
        help="stop an already-running neurodash and start a fresh instance")
    args = parser.parse_args()

    if _server_running():
        if not args.restart:
            print(f"neurodash is already running at {URL} — opening it.\n"
                  f"(Use `neurodash --restart` to stop it and start fresh.)")
            webbrowser.open(URL)
            return
        if not _stop_running_instance():
            print("Opening the existing instance instead.")
            webbrowser.open(URL)
            return

    _run()
