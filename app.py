"""neurodash — Dash entry point."""

from dash import Dash

from neurodash.layout import make_layout
import neurodash.callbacks  # noqa: F401 — registers callbacks

app = Dash(__name__, suppress_callback_exceptions=True)
app.layout = make_layout()

if __name__ == "__main__":
    app.run(debug=False)
