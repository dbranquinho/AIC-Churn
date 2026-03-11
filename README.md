# Customer Churn Prediction — Kaggle S6E3 Advanced XGBoost Pipeline

<hr>

![](https://img.shields.io/badge/python-3.12-lightblue) ![](https://img.shields.io/badge/XGBoost-189FDD?style=flat) ![](https://img.shields.io/badge/Optuna-20232A?style=flat) ![](https://img.shields.io/badge/Scikit--Learn-F7931E?style=flat) ![](https://img.shields.io/badge/Licence-MIT-lightgray) ![](https://img.shields.io/badge/status-Release-darkgreen)

A highly optimized machine learning pipeline developed to tackle the **[Kaggle Playground Series S6E3 Telecom Churn Dataset](https://www.kaggle.com/competitions/playground-series-s6e3)**.

Our final solution relies on an **Extreme XGBoost v3 Architecture** that utilizes advanced feature engineering, $K$-Means clustering, smoothed Out-Of-Fold (OOF) target encoding, and rigorous Optuna hyperparameter tuning to achieve exceptional ROC-AUC scores ($~0.916+$).

## Advanced Feature Engineering (The v3 Edge)

To extract maximum signal from the categorical and continuous variables, the pipeline (`src/kaggle_train.py`) automatically generates powerful synthetic features:

1. **Financial Ratios:** Ratios like `MonthlyCharges / TotalCharges`, `tenure / MonthlyCharges`, and the percentage discrepancy between actual vs. expected total charges.
2. **K-Means Financial Clustering:** Clusters the customer base ($k=8$) dynamically based on their financial profiles, providing XGBoost with explicit boundaries to optimize leaf splits.
3. **Smoothed Target Encoding (OOF):** Applies heavily regularized, noise-injected Out-Of-Fold target encoding to high-cardinality features to prevent data leakage while capturing the historical average churn rates.
4. **Feature Crosses & Binning:** Interaction terms (e.g. `Contract + PaymentMethod`), frequency counts, and discrete quantile binning for continuous features.

## Ultimate 5-Fold Ensembling

Rather than relying on a single model, the script implements a **5-Fold Stratified Ensembling strategy**. It trains 5 separate, heavily-tuned XGBoost models on different subsets of the data and averages their final predictions. This drastically reduces model variance and prevents overfitting to the public leaderboard.

## Project Structure

```text
AIC-Churn/
├── data/
│   ├── train.csv                    # Kaggle S6E3 training data (594k rows)
│   ├── test.csv                     # Kaggle S6E3 test data (254k rows)
│   └── sample_submission.csv
├── models/
│   ├── kaggle/                      # Saved 5-Fold XGBoost JSON models
│   └── plots/
│       └── xgb_v3_feature_importance.png  # Automatically generated importance plot
├── src/
│   ├── config.py                    # Hyperparameters and data paths
│   ├── dataset.py                   # Data ingestion 
│   └── kaggle_train.py              # The Ultimate v3 Training & Inference Pipeline
├── submissions/
│   └── submission_xgb_v3_ensemble.csv # Final predictions for Kaggle
├── xgb_v3_report.md                 # Auto-generated performance report
├── requirements.txt
└── README.md
```

## Setup & Installation

Ensure you have Python 3.12 installed.

```bash
pip install -r requirements.txt
```

*(Note: The `kaggle_train.py` script requires a modern XGBoost version (>=2.0) with native categorical support and Optuna for hyperparameter tuning).*

## How to Run the Pipeline

The codebase has been refactored into a **single, unified command**. Running this script will automatically load the data, engineer the v3 features, execute an 80-trial Optuna optimization, perform 5-Fold Training, and generate your Kaggle submission CSV.

```bash
python src/kaggle_train.py
```

### What the script executes sequentially:

1. **Data Preprocessing & Feature Engineering:** Applies KMeans, binning, and cross-features.
2. **Phase 1 (Optuna Tuning):** Runs 80 fast trials on GPU (`device='cuda'`) sweeping massive search spaces, ensuring learning rates are bounded (`0.005` to `0.05`) for high precision.
3. **Phase 2 (Ensembling):** Iterates through 5 Stratified Folds, applying **Smoothed OOF Target Encoding** per fold, and trains a highly robust XGBoost model (`num_boost_round=3500`).
4. **Submission Generation:** Averages the probabilities across all 5 folds and outputs `submissions/submission_xgb_v3_ensemble.csv`.
5. **Reporting:** Calculates overall OOF ROC-AUC, aggregates feature weights, saves the importance plot to `models/plots/`, and writes the `xgb_v3_report.md`.

## License

MIT
