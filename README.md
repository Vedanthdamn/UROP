# Privacy-Preserving Healthcare Learning with Federated and SplitFed Pipelines

## Project Overview

This repository implements a research-oriented healthcare machine learning system for binary heart-disease prediction under strict privacy constraints. It compares three training paradigms on the same dataset family and evaluation protocol:

- Centralized learning
- Federated Learning (Flower + FedProx)
- SplitFed (split learning + federated aggregation)

The codebase is organized for reproducibility, modularity, and benchmark-style experimentation.

## Problem Statement

Healthcare institutions often cannot pool patient records due to privacy, compliance, and ownership constraints. Standard centralized ML workflows are therefore difficult to deploy in real-world medical settings.

The challenge is to train accurate models while keeping raw hospital data local.

## Solution Summary

The project uses distributed learning strategies that avoid raw-data exchange:

- Federated training performs local updates on each hospital partition and aggregates model parameters centrally.
- SplitFed further partitions the model into client-side and server-side segments so clients send activations rather than full feature records.
- A shared privacy-safe data pipeline enforces client-local scaling and strict held-out test evaluation.

## Architecture (Text Description)

1. Raw tabular data is loaded, validated, encoded, and expanded into a reproducible training corpus.
2. Non-IID hospital partitions are generated for five clients.
3. For each hospital, train/validation/test splits are created independently and scaled using train-only statistics.
4. Centralized pipeline trains a full model on the combined train split and evaluates on a strictly unseen global test split.
5. Federated pipeline runs FedProx rounds over hospital clients and evaluates the final global model on the same global test split.
6. SplitFed pipeline trains split client/server components per round, aggregates client-side weights, and evaluates on the same global test split.
7. Metrics and research artifacts are saved to `data/processed` and `plots`.

## Tech Stack

### Backend

- Python 3.10+
- TensorFlow / Keras
- Flower (FL simulation)
- NumPy, Pandas, scikit-learn, Matplotlib
- FastAPI (metrics API)

### Frontend Dashboard

- React (Vite)
- Tailwind CSS
- Recharts
- Framer Motion

## How to Run

### 1. Backend Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Training Pipelines

```bash
python src/pipelines/train_eval_pipeline.py
python src/pipelines/federated_pipeline.py
python src/pipelines/splitfed_pipeline.py
```

### 3. Start Metrics API (Optional)

```bash
uvicorn src.utils.api_server:app --reload
```

### 4. Run Frontend Dashboard

```bash
cd frontend
npm install
npm run dev
```

## Results (Current Run)

| Method | Accuracy | Precision | Recall | F1 Score | Loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Centralized | 79.38% | 82.54% | 77.98% | 80.19% | 0.4618 |
| Federated (FedProx) | 79.38% | 89.50% | 69.65% | 78.34% | 2.1075 |
| SplitFed | 86.00% | 89.77% | 83.35% | 86.44% | 0.3347 |

Metrics source: `data/processed/final_metrics.json`.

## Key Insights

- SplitFed achieves the strongest overall test performance in this setup.
- Federated training remains robust under non-IID partitions but shows a precision-recall tradeoff.
- Strict leakage controls (train-only scaling per client and held-out test evaluation) provide fairer model comparison.

## Project Structure

```text
src/
	data/
	federated/
	models/
	pipelines/
	utils/
data/
	raw/
	processed/
clients/
plots/
frontend/
docs/
models/
```

## Future Scope

- Add formal differential privacy accounting and privacy budgets.
- Add secure aggregation and encrypted transport validation.
- Extend evaluation with calibration, subgroup fairness, and uncertainty metrics.
- Package training as reproducible experiment configs (for sweep automation).

## Author

- Vedanth Dama

## Development Note

Developed using ~20-25 structured prompt iterations.
