# Development History

## Scope

- Data preprocessing and augmentation
- Non-IID hospital dataset generation
- Centralized model training and evaluation
- Federated learning with Flower
- SplitFed hybrid pipeline
- Frontend analytics dashboard

## Quality Controls

- Leakage-free evaluation: original records are split into disjoint train/test sets
  before augmentation, so held-out patients never leak into training.
- LayerNormalization for aggregation-safe normalization across non-IID clients.
- Deterministic runs via fixed seeds and TensorFlow op-determinism.
- Frontend fallback loading from local JSON files.
