# Anvil Feedback Publication Repository

Code to reproduce figures and key numbers from Deutloff et al. (2026).

## Data
Download the required input data from Zenodo:

- https://doi.org/10.5281/zenodo.19245784

After downloading, edit the `get_path()` function in `src/read_data.py` to point to your local data directory.

## Python Environment
Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Reproduce Figures
Run scripts from the repository root. Diagnostic scripts should be run before
the plotting scripts when the intermediate slope and feedback files need to be
regenerated:

```bash
mkdir -p plots
```

```bash
python scripts/diagnose/calc_slopes.py
python scripts/diagnose/calc_feedback.py
python scripts/plot/cre_sensitivity.py
python scripts/plot/era5.py
python scripts/plot/feedback.py
python scripts/plot/hists.py
python scripts/plot/slopes.py
python scripts/plot/temp_pattern.py
```

Outputs are written to `plots/`.
