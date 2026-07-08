# Reproducibility Guide

## Backend

1. Create a Python 3.10 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run pipelines in order (SplitFed reuses the centralized and federated models for its
   comparison plots, so run it last):

```bash
python src/pipelines/train_eval_pipeline.py
python src/pipelines/federated_pipeline.py
python src/pipelines/splitfed_pipeline.py
```

Runs are deterministic (fixed seeds + TensorFlow op-determinism) and reproduce the same
metrics on re-run. The train/test split is performed on the original records before any
augmentation, so the held-out test set is leakage-free.

4. Regenerate all report figures from the trained models and saved metrics:

```bash
python src/utils/generate_report_plots.py
```

5. (Optional) Run the DP-SGD privacy ablation and regenerate its figure. Epsilon is
   computed with a Renyi-DP accountant (`src/federated/dp_accountant.py`) and is
   deterministic for a fixed seed, noise multiplier, and step count:

```bash
python src/pipelines/dp_epsilon_sweep.py
python src/utils/generate_dp_plots.py
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard reads local JSON metrics from `frontend/public/data/processed` and includes safe fallback values when files are missing.
