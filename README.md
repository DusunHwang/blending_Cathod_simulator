# Blending Cathode Simulator

This repository contains two main utilities:

1. **simulate_blend.py** – prototype PyBaMM script for a blended NCM/LFP cathode.
2. **train_regressor.py** – simple regression example used for automated testing.

Datasets for the regression example live in `./data/`. Running
`python train_regressor.py` will train a toy model and output logs in
`./logs`, plots in `./plots`, a report in `./reports`, and the model in
`./models`.

Run tests with:

```bash
pytest -q
```
