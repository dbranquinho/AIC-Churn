"""
V15 Pipeline - The 0.919+ Barrier Breaker
Implements Bi/Tri Categorical Combinations and a Level 2 Stacking Ensemble.
No pseudo-labeling data leakage.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import json
import warnings

warnings.filterwarnings('ignore')

import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_FOLDS = 10

def build_combinations(df, cat_cols):
    """
    Creates pairwise and triplet combinations to capture non-linear category interactions.
    These will be heavily smoothed via TargetEncoder later.
    """
    print("Building Bi/Tri Categorical Combinations...")
    # Known high-interaction pairs
    pairs = [
        ('Contract', 'PaymentMethod'),
        ('InternetService', 'OnlineSecurity'),
        ('InternetService', 'gender'),
        ('Partner', 'Dependents'),
        ('Contract', 'PaperlessBilling'),
        ('MultipleLines', 'InternetService'),
        ('TechSupport', 'DeviceProtection'),
        ('PaymentMethod', 'PaperlessBilling')
    ]
    
    new_cols = []
    for col1, col2 in pairs:
        new_col = f"{col1}_{col2}"
        df[new_col] = df[col1] + "_" + df[col2]
        new_cols.append(new_col)
        
    # Example Triplet
    triplet_col = 'Contract_InternetService_PaymentMethod'
    df[triplet_col] = df['Contract'] + "_" + df['InternetService'] + "_" + df['PaymentMethod']
    new_cols.append(triplet_col)
    
    return df, new_cols

def feature_engineering(train_df, test_df):
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    cat_cols = config.KAGGLE_CATEGORICAL_COLS.copy()
    
    # Generate bi/tri combinations
    df_all, new_cat_cols = build_combinations(df_all, cat_cols)
    cat_cols.extend(new_cat_cols)

    # Revert to fillna(0) for stability
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
    # The Grandmaster's "ChargeResidual" Trick
    expected_charge = df_all['MonthlyCharges'] * df_all['tenure']
    charge_residual = df_all['TotalCharges'] - expected_charge
    
    df_all['charge_residual'] = charge_residual
    df_all['charge_residual_sign'] = np.sign(charge_residual)
    df_all['charge_residual_relative'] = charge_residual / (expected_charge + 1)
    
    # Standard Logical Ratios
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + 1)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + 1)
    
    # Fill categorical nulls
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    print(f"Final Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y, X_test, ids, cat_cols


def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, cat_cols = feature_engineering(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  V15 Top-Tier Training ({NUM_FOLDS}-Fold Stacking)")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # OOF probabilities storage logic [XGB, LGBM, CAT]
    oof_preds = np.zeros((len(X_train), 3))
    test_preds_l1 = np.zeros((len(X_test), 3))
    
    # Model parameters
    best_te_smoothing = 40.0
    
    xgb_p = {
      "learning_rate": 0.03,
      "max_depth": 6,
      "subsample": 0.85,
      "colsample_bytree": 0.30,
      "min_child_weight": 18,
      "gamma": 0.003,
      "reg_alpha": 0.11,
      "reg_lambda": 12.0,
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "tree_method": "hist",
      "random_state": 42
    }
    
    lgb_p = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.02,
        'num_leaves': 31,
        'max_depth': 6,
        'feature_fraction': 0.7,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1,
        'seed': 42
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        # 1. Target Encoding
        te = TargetEncoder(cols=cat_cols, smoothing=best_te_smoothing)
        X_tr_enc = te.fit_transform(X_tr, y_tr)
        X_va_enc = te.transform(X_va)
        X_te_enc = te.transform(X_test)
        
        # 2. Train XGBoost
        dtrain_xgb = xgb.DMatrix(X_tr_enc, label=y_tr)
        dvalid_xgb = xgb.DMatrix(X_va_enc, label=y_va)
        dtest_xgb = xgb.DMatrix(X_te_enc)
        
        m_xgb = xgb.train(xgb_p, dtrain_xgb, 2500, evals=[(dvalid_xgb, 'valid')], early_stopping_rounds=150, verbose_eval=False)
        oof_preds[val_idx, 0] = m_xgb.predict(dvalid_xgb)
        test_preds_l1[:, 0] += m_xgb.predict(dtest_xgb) / NUM_FOLDS
        
        # 3. Train LightGBM
        dtrain_lgb = lgb.Dataset(X_tr_enc, label=y_tr)
        dvalid_lgb = lgb.Dataset(X_va_enc, label=y_va)
        
        m_lgb = lgb.train(lgb_p, dtrain_lgb, num_boost_round=2500, valid_sets=[dvalid_lgb], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds[val_idx, 1] = m_lgb.predict(X_va_enc, num_iteration=m_lgb.best_iteration)
        test_preds_l1[:, 1] += m_lgb.predict(X_te_enc, num_iteration=m_lgb.best_iteration) / NUM_FOLDS
        
        # 4. Train CatBoost
        m_cat = CatBoostClassifier(iterations=2500, learning_rate=0.03, depth=6, eval_metric='AUC', random_seed=42, verbose=False, early_stopping_rounds=150)
        m_cat.fit(X_tr_enc, y_tr, eval_set=(X_va_enc, y_va))
        oof_preds[val_idx, 2] = m_cat.predict_proba(X_va_enc)[:, 1]
        test_preds_l1[:, 2] += m_cat.predict_proba(X_te_enc)[:, 1] / NUM_FOLDS
        
        print(f"Fold {fold+1} Individual OOF AUCs => XGB: {roc_auc_score(y_va, oof_preds[val_idx, 0]):.5f} | LGB: {roc_auc_score(y_va, oof_preds[val_idx, 1]):.5f} | CAT: {roc_auc_score(y_va, oof_preds[val_idx, 2]):.5f}")

    # ==========================================
    # LEVEL 2: Stacking Meta-Model
    # ==========================================
    print("\n" + "="*60)
    print("  Phase 2: Training Level-2 Meta Model (Logistic Regression)")
    print("="*60)
    
    meta_model = LogisticRegression()
    # Fit the meta model on the entire Out-Of-Fold predictions
    meta_model.fit(oof_preds, y)
    
    final_oof_preds = meta_model.predict_proba(oof_preds)[:, 1]
    final_auc = roc_auc_score(y, final_oof_preds)
    
    print("\n" + "="*60)
    print(f"  V15 Bi/Tri Stacking FINAL True OOF AUC: {final_auc:.5f}")
    print("="*60)
    print(f"Meta-Model Coefficients [XGB, LGBM, CAT]: {meta_model.coef_[0]}")
    
    # Predict final answers on Test
    final_test_preds = meta_model.predict_proba(test_preds_l1)[:, 1]
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': final_test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v15_stack.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    print(f"\nDONE! Submission saved to: {sub_prob_path}")

if __name__ == "__main__":
    main()
