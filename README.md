# neurodash
<img src="https://raw.githubusercontent.com/EricThomson/neurodash/main/src/neurodash/assets/logo/neurodash_logo_trimmed.png" alt="neurodash logo" align="right" width="250">
Lightweight dashboard for exploratory analysis of neurobehavioral data.

## Usage
Install uv locally following the instructions here:    
    
    https://docs.astral.sh/uv/getting-started/installation/

Install neurodash:    

    uv tool install neurodash

Then, run neurodash by entering `neurodash` at the command line. 

To update to latest version of neurodash:    

    uv tool upgrade neurodash

## Development
In your cli:

    git clone https://github.com/EricThomson/neurodash
    cd neurodash
    uv sync
    # activate environment (e.g.,  source .venv/Scripts/activate)
    # run dashboard
    python app.py

## Roadmap
- Arena calibration scaling.
- docs
  - basic how to use
  - Screenshots or gifs
  - explain back-end stuff like channels/streaming neo/plx
- Testing, linting, etc
  