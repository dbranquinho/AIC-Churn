# V21 LightGBM Shallow Native

## Final Performance
- **10-Fold OOF AUC**: **0.91544**

## Pipeline Details
Stripped all engineered features. Ran pure LightGBM natively with `max_depth=3`, `min_child_samples=100`, and extreme L1/L2=5.0 to suppress synthetic noise absorption.
