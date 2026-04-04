# Reproducibility Guide

## Backend

1. Create a Python 3.10 environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run pipelines:

```bash
python src/pipelines/train_eval_pipeline.py
python src/pipelines/federated_pipeline.py
python src/pipelines/splitfed_pipeline.py
```

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The dashboard reads local JSON metrics from `frontend/public/data/processed` and includes safe fallback values when files are missing.
