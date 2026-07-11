"""
launch.py — entry point for the neurodash PyQt viewer.

Reads a handoff JSON file written by the Dash app, then launches
NeurodashViewer. Called as:

    python -m neurodash.pyqtdash.launch --handoff /path/to/handoff.json
"""

import argparse
import sys

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication

from neurodash.pyqtdash.main_window import NeurodashViewer


def main():
    parser = argparse.ArgumentParser(description="neurodash PyQt viewer")
    parser.add_argument("--handoff", required=True, help="Path to handoff directory")
    args = parser.parse_args()

    app = QApplication(sys.argv)
    pg.setConfigOption("background", "k")
    pg.setConfigOption("foreground", "w")
    win = NeurodashViewer(args.handoff)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
