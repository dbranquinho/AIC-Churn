"""
Kaggle Playground Series S6E3 -- Telecom Churn Training Pipeline
Trains XGBoost, LightGBM, CatBoost, and PyTorch MLP with Stratified 5-Fold CV.
"""
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from tqdm import tqdm

import src.config as config
from src.dataset import kaggle_load_data, KaggleDataProcessor, ChurnDataset, get_dataloader
from src.model import ChurnModel

# Output paths
KAGGLE_MODEL_DIR = os.path.join(config.MODEL_DIR, 'kaggle')
os.makedirs(KAGGLE_MODEL_DIR, exist_ok=True)

PROCESSOR_PATH = os.path.join(KAGGLE_MODEL_DIR, 'processor.pkl')
XGB_PATH = os.path.join(KAGGLE_MODEL_DIR, 'xgb_model.json')
LGB_PATH = os.path.join(KAGGLE_MODEL_DIR, 'lgb_model.txt')
CAT_PATH = os.path.join(KAGGLE_MODEL_DIR, 'cat_model.cbm')
MLP_PATH = os.path.join(KAGGLE_MODEL_DIR, 'mlp_model.pth')


def evaluate_fold(y_true, y_prob):
    """Return dict of metrics for a single fold."""
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'accuracy': accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred),
    }


def print_cv_results(name, fold_metrics):
    """Pretty-print cross-validation results."""
    print(f"\n{'='*60}")
    print(f"  {name} -- {config.KAGGLE_N_FOLDS}-Fold CV Results")
    print(f"{'='*60}")
    for metric in ['roc_auc', 'accuracy', 'f1']:
        values = [m[metric] for m in fold_metrics]
        print(f"  {metric:>10s}: {np.mean(values):.5f} ± {np.std(values):.5f}  "
              f"(folds: {', '.join(f'{v:.4f}' for v in values)})")
    print(f"{'='*60}\n")


# ────────────────────────────────────────────────────────────
# XGBoost
# ────────────────────────────────────────────────────────────
def train_xgboost_cv(X, y, skf):
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            objective='binary:logistic', eval_metric='auc',
            tree_method='hist', random_state=config.KAGGLE_RANDOM_STATE,
            n_jobs=-1, verbosity=0,
        )
        clf.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
            verbose=False,
        )
        prob = clf.predict_proba(X[va_idx])[:, 1]
        fold_metrics.append(evaluate_fold(y[va_idx], prob))
        print(f"  XGB fold {fold}: AUC={fold_metrics[-1]['roc_auc']:.5f}")
    print_cv_results("XGBoost", fold_metrics)
    return fold_metrics


def train_xgboost_full(X, y):
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        objective='binary:logistic', eval_metric='auc',
        tree_method='hist', random_state=config.KAGGLE_RANDOM_STATE,
        n_jobs=-1, verbosity=0,
    )
    clf.fit(X, y)
    clf.save_model(XGB_PATH)
    print(f"  XGBoost full model saved -> {XGB_PATH}")
    return clf


# ────────────────────────────────────────────────────────────
# LightGBM
# ────────────────────────────────────────────────────────────
def train_lightgbm_cv(X, y, skf):
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = lgb.LGBMClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
            random_state=config.KAGGLE_RANDOM_STATE, n_jobs=-1, verbose=-1,
        )
        clf.fit(
            X[tr_idx], y[tr_idx],
            eval_set=[(X[va_idx], y[va_idx])],
        )
        prob = clf.predict_proba(X[va_idx])[:, 1]
        fold_metrics.append(evaluate_fold(y[va_idx], prob))
        print(f"  LGB fold {fold}: AUC={fold_metrics[-1]['roc_auc']:.5f}")
    print_cv_results("LightGBM", fold_metrics)
    return fold_metrics


def train_lightgbm_full(X, y):
    clf = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        random_state=config.KAGGLE_RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    clf.fit(X, y)
    clf.booster_.save_model(LGB_PATH)
    print(f"  LightGBM full model saved -> {LGB_PATH}")
    return clf


# ────────────────────────────────────────────────────────────
# CatBoost
# ────────────────────────────────────────────────────────────
def train_catboost_cv(X, y, skf):
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        clf = CatBoostClassifier(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=3.0, loss_function='Logloss', eval_metric='AUC',
            random_seed=config.KAGGLE_RANDOM_STATE, verbose=0,
        )
        clf.fit(
            X[tr_idx], y[tr_idx],
            eval_set=(X[va_idx], y[va_idx]),
        )
        prob = clf.predict_proba(X[va_idx])[:, 1]
        fold_metrics.append(evaluate_fold(y[va_idx], prob))
        print(f"  CAT fold {fold}: AUC={fold_metrics[-1]['roc_auc']:.5f}")
    print_cv_results("CatBoost", fold_metrics)
    return fold_metrics


def train_catboost_full(X, y):
    clf = CatBoostClassifier(
        iterations=500, depth=6, learning_rate=0.05,
        l2_leaf_reg=3.0, loss_function='Logloss', eval_metric='AUC',
        random_seed=config.KAGGLE_RANDOM_STATE, verbose=0,
    )
    clf.fit(X, y)
    clf.save_model(CAT_PATH)
    print(f"  CatBoost full model saved -> {CAT_PATH}")
    return clf


# ────────────────────────────────────────────────────────────
# PyTorch MLP
# ────────────────────────────────────────────────────────────
def _train_mlp_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(batch_X), batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def _predict_mlp(model, X, device, batch_size=2048):
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


def train_mlp_cv(X, y, skf, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        model = ChurnModel(
            input_dim=input_dim,
            hidden_units=config.KAGGLE_HIDDEN_UNITS,
            dropout_rate=config.KAGGLE_DROPOUT_RATE,
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=config.KAGGLE_LEARNING_RATE)

        train_loader = get_dataloader(
            X[tr_idx], y[tr_idx], config.KAGGLE_BATCH_SIZE, shuffle=True
        )

        best_auc = 0
        patience_counter = 0
        for epoch in range(config.KAGGLE_EPOCHS):
            avg_loss = _train_mlp_epoch(model, train_loader, criterion, optimizer, device)
            prob = _predict_mlp(model, X[va_idx], device)
            auc = roc_auc_score(y[va_idx], prob)
            if auc > best_auc:
                best_auc = auc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 3:
                    break

        prob = _predict_mlp(model, X[va_idx], device)
        fold_metrics.append(evaluate_fold(y[va_idx], prob))
        print(f"  MLP fold {fold}: AUC={fold_metrics[-1]['roc_auc']:.5f}")
    print_cv_results("PyTorch MLP", fold_metrics)
    return fold_metrics


def train_mlp_full(X, y, input_dim):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChurnModel(
        input_dim=input_dim,
        hidden_units=config.KAGGLE_HIDDEN_UNITS,
        dropout_rate=config.KAGGLE_DROPOUT_RATE,
    ).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.KAGGLE_LEARNING_RATE)
    train_loader = get_dataloader(X, y, config.KAGGLE_BATCH_SIZE, shuffle=True)

    for epoch in range(config.KAGGLE_EPOCHS):
        avg_loss = _train_mlp_epoch(model, train_loader, criterion, optimizer, device)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    Epoch {epoch+1}/{config.KAGGLE_EPOCHS} -- loss: {avg_loss:.5f}")

    torch.save(model.state_dict(), MLP_PATH)
    print(f"  MLP full model saved -> {MLP_PATH}")
    return model


# ────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────
def main():
    start = time.time()

    # 1. Load & preprocess
    print("\n" + "="*60)
    print("  Loading & Preprocessing Training Data")
    print("="*60)
    X, y, processor = kaggle_load_data(config.KAGGLE_TRAIN_PATH, fit_processor=True)
    processor.save(PROCESSOR_PATH)
    input_dim = processor.get_feature_dim()
    print(f"  Samples: {X.shape[0]:,} | Features: {X.shape[1]} | Churn rate: {y.mean():.3%}")

    # 2. Cross-validation
    skf = StratifiedKFold(
        n_splits=config.KAGGLE_N_FOLDS,
        shuffle=True,
        random_state=config.KAGGLE_RANDOM_STATE,
    )

    print("\n" + "="*60)
    print("  Cross-Validation Phase")
    print("="*60)

    all_results = {}
    all_results['XGBoost'] = train_xgboost_cv(X, y, skf)
    all_results['LightGBM'] = train_lightgbm_cv(X, y, skf)
    all_results['CatBoost'] = train_catboost_cv(X, y, skf)
    all_results['MLP'] = train_mlp_cv(X, y, skf, input_dim)

    # 3. Full training
    print("\n" + "="*60)
    print("  Full Training Phase (all data)")
    print("="*60)
    train_xgboost_full(X, y)
    train_lightgbm_full(X, y)
    train_catboost_full(X, y)
    train_mlp_full(X, y, input_dim)

    # 4. Summary
    elapsed = time.time() - start
    print("\n" + "="*60)
    print("  FINAL SUMMARY")
    print("="*60)
    print(f"{'Model':<15} {'AUC':>10} {'Accuracy':>10} {'F1':>10}")
    print("-" * 48)
    for name, folds in all_results.items():
        auc = np.mean([m['roc_auc'] for m in folds])
        acc = np.mean([m['accuracy'] for m in folds])
        f1 = np.mean([m['f1'] for m in folds])
        print(f"{name:<15} {auc:>10.5f} {acc:>10.5f} {f1:>10.5f}")
    print(f"\nTotal time: {elapsed:.1f}s")
    print(f"Models saved in: {KAGGLE_MODEL_DIR}")


if __name__ == "__main__":
    main()
