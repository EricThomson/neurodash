"""neurodash — dev entry point.

`python app.py` is kept for local development; it does exactly what the installed
`neurodash` command does (launch the dashboard and open a browser).
"""

from neurodash.app import main

if __name__ == "__main__":
    main()
