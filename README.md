# Customer Churn Prediction — Multi-Model AI Pipeline

<hr>

![](https://img.shields.io/badge/python-3.12-lightblue) ![](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) ![](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white) ![](https://img.shields.io/badge/XGBoost-189FDD?style=flat) ![](https://img.shields.io/badge/LightGBM-02569B?style=flat) ![](https://img.shields.io/badge/CatBoost-FFCD00?style=flat) ![](https://img.shields.io/badge/Licence-MIT-lightgray) ![](https://img.shields.io/badge/status-Release-darkgreen)

A production-ready churn prediction system that compares **6 algorithms** (2 deep learning + 3 gradient boosters + 1 ensemble), includes a **deep diagnostic evaluation suite** with 13+ advanced metrics, and provides mathematical proof of dataset shift via adversarial validation and per-feature PSI analysis.

## Technology Stack

| Category | Technologies |
|---|---|
| **Deep Learning** | [PyTorch](https://pytorch.org/) (MLP), [TensorFlow/Keras](https://www.tensorflow.org/) (MLP) |
| **Gradient Boosting** | [XGBoost](https://xgboost.readthedocs.io/), [LightGBM](https://lightgbm.readthedocs.io/), [CatBoost](https://catboost.ai/) |
| **Preprocessing** | [Scikit-Learn](https://scikit-learn.org/) (`StandardScaler`, `OneHotEncoder`, `ColumnTransformer`) |
| **Data** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) |
| **Visualization** | [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/) |

## Project Structure

```text
AIC-Churn/
├── data/
│   ├── customer_churn_dataset-training-master.csv   # ~440K training records
│   └── customer_churn_dataset-testing-master.csv     # ~64K testing records
├── models/                          # Saved trained models & processors
│   ├── churn_model.pth              # PyTorch MLP weights
│   ├── tf_churn_model.keras         # TensorFlow/Keras model
│   ├── xgb_model.json               # XGBoost model
│   ├── lgb_model.txt                # LightGBM model
│   ├── cat_model.cbm                # CatBoost model
│   └── *.pkl                        # Fitted preprocessor pipelines
├── assets/                          # Generated diagnostic plots
│   ├── confusion_matrix.png
│   ├── roc_curve.png                # Comparative ROC (PyTorch vs TF)
│   ├── pr_curve.png                 # Comparative PR curve
│   ├── calibration_and_confidence.png   # Calibration + confidence histogram
│   ├── ks_gains_lift.png            # KS, Cumulative Gains, Lift (all models)
│   ├── psi_distribution_shift.png   # Per-feature train/test distributions
│   └── dca_youden.png              # Decision Curve Analysis + Youden's J
├── src/
│   ├── config.py                    # Hyperparameters and paths
│   ├── dataset.py                   # Custom Dataset and DataProcessor
│   ├── model.py                     # PyTorch MLP architecture
│   ├── model_tf.py                  # TensorFlow/Keras MLP architecture
│   ├── train.py                     # Train PyTorch MLP
│   ├── train_tf.py                  # Train TensorFlow/Keras MLP
│   ├── train_xgb.py                 # Train XGBoost
│   ├── train_lgb.py                 # Train LightGBM
│   ├── train_cat.py                 # Train CatBoost
│   ├── evaluate.py                  # Evaluate PyTorch MLP
│   ├── evaluate_tf.py               # Evaluate TensorFlow/Keras MLP
│   ├── evaluate_xgb.py              # Evaluate XGBoost
│   ├── evaluate_lgb.py              # Evaluate LightGBM
│   ├── evaluate_cat.py              # Evaluate CatBoost
│   ├── inference.py                 # Production inference on new data
│   ├── report.py                    # Generate ROC/PR/Confusion plots
│   ├── deep_evaluation.py           # Advanced 13+ metric diagnostic suite
│   ├── adversarial_validation.py    # Dataset shift proof
│   └── describe.py                  # Data exploration
├── report.md                        # Full evaluation report
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

### Dependencies

```
torch, tensorflow, pandas, numpy, scikit-learn, tqdm, matplotlib, seaborn
xgboost, lightgbm, catboost  (for gradient boosting models)
```

## Usage

### 1. Train Models

```bash
# Deep Learning
python -m src.train          # PyTorch MLP
python -m src.train_tf       # TensorFlow/Keras MLP

# Gradient Boosting
python -m src.train_xgb      # XGBoost
python -m src.train_lgb      # LightGBM
python -m src.train_cat      # CatBoost
```

### 2. Evaluate Models

```bash
python -m src.evaluate       # PyTorch MLP
python -m src.evaluate_tf    # TensorFlow/Keras MLP
python -m src.evaluate_xgb   # XGBoost
python -m src.evaluate_lgb   # LightGBM
python -m src.evaluate_cat   # CatBoost
```

### 3. Generate Visual Reports

```bash
python -m src.report         # Comparative ROC/PR/Confusion plots
```

### 4. Deep Diagnostic Evaluation (13+ Metrics)

```bash
python -m src.deep_evaluation
```

Runs all 5 models through 5 tiers of analysis:

| Tier | Metrics | What It Reveals |
|---|---|---|
| **1. Statistical Robustness** | MCC, Cohen's Kappa, Log Loss, Brier Score | Whether model beats random chance |
| **2. Probability Calibration** | Calibration Curve, ECE, Confidence Distribution | Whether predicted probabilities are meaningful |
| **3. Discriminative Power** | KS Statistic, Gini, Cumulative Gains, Lift | Ranking quality and business utility |
| **4. Distribution Shift** | PSI per feature, train vs. test density plots | Root cause of model failure |
| **5. Decision-Theoretic** | Decision Curve Analysis, Youden's J | Optimal threshold and net benefit |

### 5. Inference on New Data

See `src/inference.py` for how to load and use the model in production with new, raw customer records.

### 6. Adversarial Validation

```bash
python -m src.adversarial_validation
```

Mathematically proves whether training and testing datasets come from the same distribution.

## Results Summary

### Algorithm Shootout

| Architecture | Paradigm | Train Accuracy | Test Accuracy |
|:---|:---|:---:|:---:|
| PyTorch MLP | Deep Learning | 99.98% | 51.58% |
| TensorFlow/Keras MLP | Deep Learning | 98.74% | 51.60% |
| Random Forest | Bagging | 99.90% | 51.00% |
| XGBoost | Gradient Boosting | 99.98% | 50.35% |
| LightGBM | Gradient Boosting | 99.98% | 50.34% |
| CatBoost | Gradient Boosting | 99.99% | 50.34% |

All models cap at ~50% test accuracy due to **Covariate Shift** — confirmed by adversarial validation (AUC = 0.76) and PSI analysis identifying `Support Calls` (PSI=0.38) and `Payment Delay` (PSI=0.30) as the root cause.

### Key Advanced Metrics (All Models)

| Metric | PyTorch | TF/Keras | XGBoost | LightGBM | CatBoost |
|:---|:---:|:---:|:---:|:---:|:---:|
| Accuracy | 0.5158 | 0.5160 | 0.5035 | 0.5034 | 0.5034 |
| ROC AUC | 0.5973 | 0.5997 | **0.7446** | **0.7450** | 0.6306 |
| MCC | 0.1904 | 0.1901 | 0.1625 | 0.1623 | 0.1623 |
| KS Statistic | 0.1731 | 0.1789 | **0.3984** | **0.3987** | 0.2332 |
| Gini | 0.1947 | 0.1993 | **0.4892** | **0.4900** | 0.2611 |

> See [report.md](report.md) for the complete analysis with 13 metrics, 7 diagnostic plots, and strategic conclusions.

## License

MIT
