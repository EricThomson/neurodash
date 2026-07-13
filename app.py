"""neurodash — Dash entry point."""

from pathlib import Path

from dash import Dash

import neurodash
from neurodash.layout import make_layout
import neurodash.callbacks  # noqa: F401 — registers callbacks

# Assets live inside the package (src/neurodash/assets) so they ship with a
# pip install and resolve regardless of CWD; point Dash at them explicitly.
_ASSETS = str(Path(neurodash.__file__).parent / "assets")

app = Dash(__name__, assets_folder=_ASSETS, suppress_callback_exceptions=True)
app.layout = make_layout()

if __name__ == "__main__":
    app.run(debug=False)
