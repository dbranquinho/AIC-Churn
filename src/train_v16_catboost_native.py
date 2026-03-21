"""
V16 Pipeline - The CatBoost Unleashed (Fast Edition)
Bypasses TargetEncoder entirely. Feeds raw String combinations directly
to CatBoost to leverage its native, highly-optimized categorical tree builder.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import json
import warnings

warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier, Pool

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_FOLDS = 10

def feature_engineering(train_df, test_df):
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    cat_cols = config.KAGGLE_CATEGORICAL_COLS.copy()
    
    # Revert to fillna(0) for stability
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
    expected_charge = df_all['MonthlyCharges'] * df_all['tenure']
    charge_residual = df_all['TotalCharges'] - expected_charge
    
    df_all['charge_residual'] = charge_residual
    df_all['charge_residual_sign'] = np.sign(charge_residual).astype(str) # Feed sign as Categorical!
    cat_cols.append('charge_residual_sign')
    df_all['charge_residual_relative'] = charge_residual / (expected_charge + 1)
    
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + 1)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + 1)
    
    # Fill categorical nulls WITH STRING
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    print(f"Final Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Total Categorical Columns passed natively to CatBoost: {len(cat_cols)}")
    return X_train, y, X_test, ids, cat_cols


def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, cat_cols = feature_engineering(train_df, test_df)
    
    best_p = {
        'iterations': 2000,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.5,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'GPU'
    }
    
    print("\n" + "="*60)
    print(f"  Final {NUM_FOLDS}-Fold V16 CatBoost (No Optuna, direct baseline)")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    test_pool = Pool(X_test, cat_features=cat_cols)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
        
        model = CatBoostClassifier(**best_p)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=150, verbose=200)
        
        oof_preds[val_idx] = model.predict_proba(valid_pool)[:, 1]
        test_preds += model.predict_proba(test_pool)[:, 1] / NUM_FOLDS
        print(f"Fold {fold+1} OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")
        
    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*60)
    print(f"  Final V16 CatBoost OOF AUC: {final_auc:.5f} !!!")
    print("="*60)
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v16_catboost.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    with open(os.path.join(config.BASE_DIR, 'catboost_v16_report.md'), 'w') as f:
        f.write(f"# V16 CatBoost Native\n\n## Final Performance\n- **10-Fold OOF AUC**: **{final_auc:.5f}**\n\n## Top Best Parameters Found\n```json\n{json.dumps(best_p, indent=2)}\n```\n")
    
    print("\nDONE! Submission saved.")

if __name__ == "__main__":
    main()
