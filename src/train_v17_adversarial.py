"""
V17 Pipeline - Deep Research: Adversarial Validation & Feature Pruning
Instead of blindly adding features, this script detects distribution shift 
between Train and Test (which is common in Kaggle synthetic datasets).
It trains an adversarial classifier to distinguish Train vs Test.
Features that the model uses to tell them apart are causing "Covariate Shift" 
and leading to overfitting on OOF while failing on the Leaderboard.
This script automatically drops the most drifting features before training the final model.
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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_FOLDS = 5 # Faster for adversarial validation checking
NUM_FOLDS_FINAL = 10

def feature_engineering(train_df, test_df):
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    # Mark which dataset is which for Adversarial Validation
    train_df['is_test'] = 0
    test_df['is_test'] = 1
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    cat_cols = config.KAGGLE_CATEGORICAL_COLS.copy()
    
    # Base numeric cleaning
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
    # Keep the V16 features, we want to test if they are actually causing drift!
    expected_charge = df_all['MonthlyCharges'] * df_all['tenure']
    charge_residual = df_all['TotalCharges'] - expected_charge
    
    df_all['charge_residual'] = charge_residual
    df_all['charge_residual_sign'] = np.sign(charge_residual).astype(str)
    cat_cols.append('charge_residual_sign')
    df_all['charge_residual_relative'] = charge_residual / (expected_charge + 1)
    
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + 1)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + 1)
    
    # Fill categorical nulls
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        
    y_adversarial = df_all['is_test'].values
    df_all = df_all.drop(columns=['is_test'])
    
    return df_all, y, y_adversarial, train_len, ids, cat_cols

def adversarial_validation(df_all, y_adv, cat_cols):
    print("\n" + "="*60)
    print("  PHASE 1: ADVERSARIAL VALIDATION (Train vs Test)")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(df_all))
    
    feature_importances = np.zeros(df_all.shape[1])
    
    # Fast parameters for adversarial check
    adv_params = {
        'iterations': 500,
        'learning_rate': 0.05,
        'depth': 4,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'CPU' # CPU is fine for fast 500 iter trees
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_all, y_adv)):
        X_tr, y_tr = df_all.iloc[train_idx], y_adv[train_idx]
        X_va, y_va = df_all.iloc[val_idx], y_adv[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
        
        model = CatBoostClassifier(**adv_params)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=50, verbose=False)
        
        oof_preds[val_idx] = model.predict_proba(valid_pool)[:, 1]
        feature_importances += model.get_feature_importance() / NUM_FOLDS
        
    adv_auc = roc_auc_score(y_adv, oof_preds)
    print(f"Adversarial AUC: {adv_auc:.5f}")
    
    if adv_auc > 0.52:
        print("WARNING: Distribution Shift Detected! The model can distinguish Train from Test.")
    else:
        print("Great! Train and Test distributions are very similar.")
        
    # Analyze Feature Importances for Drift
    feat_imp_df = pd.DataFrame({
        'Feature': df_all.columns,
        'Adversarial_Importance': feature_importances
    }).sort_values(by='Adversarial_Importance', ascending=False)
    
    print("\nTop 5 Most Drifting Features:")
    print(feat_imp_df.head(5).to_string(index=False))
    
    # Identify features to drop (those contributing heavily to the drift)
    # Threshold: anything above 15% importance in the adversarial model
    drifting_features = feat_imp_df[feat_imp_df['Adversarial_Importance'] > 15.0]['Feature'].tolist()
    
    if len(drifting_features) > 0:
        print(f"\n[ACTION] Dropping highly drifting features to prevent overfitting: {drifting_features}")
    else:
        print("\n[ACTION] No single feature is dominating the drift. Keeping all features.")
        
    return drifting_features

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    df_all, y, y_adversarial, train_len, test_ids, cat_cols = feature_engineering(train_df, test_df)
    
    # 1. Run Adversarial Validation to find Covariate Shift
    drifting_features = adversarial_validation(df_all, y_adversarial, cat_cols)
    
    # 2. Prune the features
    df_all = df_all.drop(columns=drifting_features)
    cat_cols = [c for c in cat_cols if c not in drifting_features]
    
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    print("\n" + "="*60)
    print(f"  PHASE 2: FINAL MODEL TRAINING ({NUM_FOLDS_FINAL}-Fold)")
    print("="*60)
    print(f"Features used: {len(X_train.columns)}")
    
    best_p = {
        'iterations': 2500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.5,
        'bagging_temperature': 0.5,
        'border_count': 128,
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'GPU' # Re-enable GPU for final deep training
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS_FINAL, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    test_pool = Pool(X_test, cat_features=cat_cols)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"--- Training Fold {fold+1}/{NUM_FOLDS_FINAL} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
        
        model = CatBoostClassifier(**best_p)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=150, verbose=False)
        
        oof_preds[val_idx] = model.predict_proba(valid_pool)[:, 1]
        test_preds += model.predict_proba(test_pool)[:, 1] / NUM_FOLDS_FINAL
        print(f"Fold {fold+1} OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")
        
    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*60)
    print(f"  Final V17 (Adversarial Pruned) OOF AUC: {final_auc:.5f}")
    if final_auc > 0.91757:
        print("  TARGET EXCEEDED! > 0.91757")
    print("="*60)
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v17_adversarial.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    with open(os.path.join(config.BASE_DIR, 'catboost_v17_report.md'), 'w') as f:
        f.write(f"# V17 CatBoost Adversarial Validation\n\n## Final Performance\n- **10-Fold OOF AUC**: **{final_auc:.5f}**\n\n## Dropped Drifting Features\n`{drifting_features}`\n\n")
    
    print("\nDONE! Submission saved to:", sub_prob_path)

if __name__ == "__main__":
    main()
