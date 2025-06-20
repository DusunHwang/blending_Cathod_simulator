# Blending Cathode Simulator

This repository demonstrates two workflows:

1. **simulate_blend.py** – a PyBaMM script for a blended NCM/LFP cathode using the DFN model.
2. **train_regressor.py** – a simple regression example implemented with PyTorch and scikit‑learn.

A ready‑to‑run notebook, **run_notebook.ipynb**, installs dependencies, trains the model and launches the simulation.

Datasets for the regression example live in `./data/`. Running
`python train_regressor.py` will train the model and output logs in
`./logs`, plots in `./plots`, a report in `./reports`, and the model in
`./models`.

Run tests with:

```bash
pytest -q
```
