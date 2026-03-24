"""
V21 Pipeline - True Structural Generalization (Shallow LightGBM)
History proved that any complex feature engineering (ratios, geometric distances, pair targets)
caused the models to overfit the synthetic generative noise within the specific Kaggle CTGAN splits.
This pipeline brutally removes all feature engineering and enforces generalization:
1. RAW Features ONLY.
2. LightGBM Native Categorical Handling (No OneHot, No TargetEncoding).
3. Shallow Trees (max_depth=3).
4. Heavy Regularization (L1=5.0, L2=5.0, min_child_samples=100).
This forces the trees to learn ONLY the macro-patterns of churn, ignoring synthetic micro-artifacts.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_FOLDS = 10

def prepare_raw_features(train_df, test_df):
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    # Strictly the base 19 Features - We are not doing ANY math here.
    cat_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod'
    ]
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    
    # Clean numeric features carefully
    for col in numeric_features:
        # Instead of fillna(0), let's keep NaNs for LightGBM to handle natively!
        # LightGBM is brilliant at optimizing missing values paths automatically.
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')
    
    # Handle Categoricals securely for LightGBM
    for col in cat_features:
        df_all[col] = df_all[col].fillna("Missing").astype('category')
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    return X_train, y, X_test, ids, cat_features

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, cat_features = prepare_raw_features(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  V21 EXTREME GENERALIZATION LIGHTGBM ({NUM_FOLDS}-Fold CV)")
    print("="*60)
    
    # Highly constrained hyper-parameters to prevent synthetic noise adaptation
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': 0.01,         # Very slow learning
        'max_depth': 3,                # SHALLOW TREES (Macro patterns only)
        'num_leaves': 7,               # Strictly limited leaves
        'min_child_samples': 100,      # Prevent splitting extreme noise
        'reg_alpha': 5.0,              # L1 Regularization to prune useless features
        'reg_lambda': 5.0,             # L2 Regularization
        'colsample_bytree': 0.5,       # Look at fewer features per split
        'subsample': 0.8,              # Bagging
        'subsample_freq': 1,
        'cat_smooth': 15.0,            # Smooth rare categories
        'random_state': 42,
        'verbose': -1,
        'n_estimators': 4000
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        # We don't need to specify categorical_feature in fit() if the purely pandas dtype is 'category'
        # LightGBM handles it natively.
        model = lgb.LGBMClassifier(**lgb_params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            eval_metric='auc',
            callbacks=[lgb.early_stopping(stopping_rounds=300, verbose=False)]
        )
        
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / NUM_FOLDS
        print(f"  [LGBM] Fold {fold+1} OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")

    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*80)
    print(f"  Final V21 (Generalization Pipeline) OOF AUC: {final_auc:.5f}")
    print("="*80)
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v21_lightgbm_shallow.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    with open(os.path.join(config.BASE_DIR, 'lightgbm_v21_report.md'), 'w') as f:
        f.write(f"# V21 LightGBM Shallow Native\n\n## Final Performance\n- **10-Fold OOF AUC**: **{final_auc:.5f}**\n\n## Pipeline Details\nStripped all engineered features. Ran pure LightGBM natively with `max_depth=3`, `min_child_samples=100`, and extreme L1/L2=5.0 to suppress synthetic noise absorption.\n")
    
    print("\nDONE! Submission saved to:", sub_prob_path)
    print("WARNING: This OOF AUC will likely be significantly LOWER than your V20 OOF score, but the Public Leaderboard test score might be HIGHER because it actually generalizes!")

if __name__ == "__main__":
    main()
