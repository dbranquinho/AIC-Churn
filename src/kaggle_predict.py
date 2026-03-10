"""
Kaggle Playground Series S6E3 -- Submission Generator
Loads trained models, predicts on test.csv, creates per-model and ensemble submissions.
"""
import os
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

import src.config as config
from src.dataset import KaggleDataProcessor, ChurnDataset
from src.model import ChurnModel

# Paths (must match kaggle_train.py)
KAGGLE_MODEL_DIR = os.path.join(config.MODEL_DIR, 'kaggle')
PROCESSOR_PATH = os.path.join(KAGGLE_MODEL_DIR, 'processor.pkl')
XGB_PATH = os.path.join(KAGGLE_MODEL_DIR, 'xgb_model.json')
LGB_PATH = os.path.join(KAGGLE_MODEL_DIR, 'lgb_model.txt')
CAT_PATH = os.path.join(KAGGLE_MODEL_DIR, 'cat_model.cbm')
MLP_PATH = os.path.join(KAGGLE_MODEL_DIR, 'mlp_model.pth')

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def predict_mlp(X, input_dim, device, batch_size=2048):
    model = ChurnModel(
        input_dim=input_dim,
        hidden_units=config.KAGGLE_HIDDEN_UNITS,
        dropout_rate=0.0,  # no dropout at inference
    ).to(device)
    model.load_state_dict(torch.load(MLP_PATH, map_location=device, weights_only=True))
    model.eval()

    dataset = ChurnDataset(X, None)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    probs = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            logits = model(batch.to(device))
            probs.append(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.concatenate(probs)


def save_submission(ids, preds, filename, label="submission"):
    """Save a submission CSV and print stats."""
    sub = pd.DataFrame({'id': ids, 'Churn': preds})
    path = os.path.join(SUBMISSION_DIR, filename)
    sub.to_csv(path, index=False)
    print(f"  [{label}] saved -> {path}  |  shape={sub.shape}  |  churn_rate={preds.mean():.3%}")
    return sub


def validate_submission(sub, sample_sub_path):
    """Validate that our submission matches the expected format."""
    sample = pd.read_csv(sample_sub_path)
    errors = []

    if list(sub.columns) != list(sample.columns):
        errors.append(f"Column mismatch: {list(sub.columns)} vs {list(sample.columns)}")
    if len(sub) != len(sample):
        errors.append(f"Row count mismatch: {len(sub)} vs {len(sample)}")
    if not set(sub['Churn'].unique()).issubset({0, 1}):
        errors.append(f"Unexpected Churn values: {sub['Churn'].unique()}")
    if not np.array_equal(sub['id'].values, sample['id'].values):
        errors.append("ID mismatch with sample_submission.csv")

    if errors:
        print("  [!] Validation FAILED:")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("  [OK] Submission format validated successfully!")
        return True


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 1. Load & preprocess test data
    print("\n" + "="*60)
    print("  Loading Test Data")
    print("="*60)
    processor = KaggleDataProcessor.load(PROCESSOR_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    ids = test_df[config.KAGGLE_ID_COL].values
    X_test = processor.transform(test_df)
    input_dim = processor.get_feature_dim()
    print(f"  Test samples: {X_test.shape[0]:,} | Features: {X_test.shape[1]}")

    # 2. Predict with each model
    print("\n" + "="*60)
    print("  Generating Predictions")
    print("="*60)
    model_probs = {}

    # XGBoost
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.load_model(XGB_PATH)
    model_probs['xgb'] = xgb_clf.predict_proba(X_test)[:, 1]
    print(f"  XGBoost predictions done -- avg prob: {model_probs['xgb'].mean():.4f}")

    # LightGBM
    lgb_booster = lgb.Booster(model_file=LGB_PATH)
    model_probs['lgb'] = lgb_booster.predict(X_test)
    print(f"  LightGBM predictions done -- avg prob: {model_probs['lgb'].mean():.4f}")

    # CatBoost
    cat_clf = CatBoostClassifier()
    cat_clf.load_model(CAT_PATH)
    model_probs['cat'] = cat_clf.predict_proba(X_test)[:, 1]
    print(f"  CatBoost predictions done -- avg prob: {model_probs['cat'].mean():.4f}")

    # MLP
    model_probs['mlp'] = predict_mlp(X_test, input_dim, device)
    print(f"  MLP predictions done -- avg prob: {model_probs['mlp'].mean():.4f}")

    # 3. Create submissions
    print("\n" + "="*60)
    print("  Saving Submissions")
    print("="*60)

    # Per-model submissions
    for name, probs in model_probs.items():
        preds = (probs >= 0.5).astype(int)
        save_submission(ids, preds, f"submission_{name}.csv", label=name.upper())

    # Ensemble (average of all probabilities)
    ensemble_prob = np.mean(list(model_probs.values()), axis=0)
    ensemble_preds = (ensemble_prob >= 0.5).astype(int)
    ensemble_sub = save_submission(ids, ensemble_preds, "submission_ensemble.csv", label="ENSEMBLE")

    # 4. Validate
    print("\n" + "="*60)
    print("  Validating Submission Format")
    print("="*60)
    validate_submission(ensemble_sub, config.KAGGLE_SAMPLE_SUB_PATH)

    print("\nDone! Upload submissions from:", SUBMISSION_DIR)


if __name__ == "__main__":
    main()
