"""
V20 Pipeline - Chris Deotte's 3x Ensemble + Hill Climbing
This matches the architecture from "ChatGPT Vibe Coding - 3xGPU Models - [CV 0.9178]".
It leverages both geometric/orthogonal decision boundaries (Trees) and pure probability
margins (Logistic Regression) on combinations of features.
1. CatBoost (GPU) trained on raw Base Features.
2. XGBoost (GPU) trained on raw Base Features.
3. Logistic Regression trained strictly on 171 Target Encoded PAIR Features.
4. SciPy Optimization (Hill Climbing variant) to find optimal AUC weights W1, W2, W3.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from category_encoders import TargetEncoder
import itertools
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

from catboost import CatBoostClassifier, Pool
import xgboost as xgb

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_FOLDS = 10

def prepare_chris_deotte_features(train_df, test_df):
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    # 1. Base 19 Features exactly
    base_features = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 
        'MonthlyCharges', 'TotalCharges'
    ]
    
    # Clean numeric features
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
    # Adaptive Binning for Pairs
    df_all['tenure_binned'] = pd.qcut(df_all['tenure'], q=15, duplicates='drop').astype(str)
    df_all['MonthlyCharges_binned'] = pd.qcut(df_all['MonthlyCharges'], q=20, duplicates='drop').astype(str)
    df_all['TotalCharges_binned'] = pd.qcut(df_all['TotalCharges'], q=20, duplicates='drop').astype(str)
    
    cat_features = [f for f in base_features if f not in ['tenure', 'MonthlyCharges', 'TotalCharges']]
    
    # Handle Tree Categorical Dtypes cleanly (Needed for XGBoost enable_categorical)
    for col in cat_features:
        df_all[col] = df_all[col].fillna("Missing").astype(str).astype('category')
        
    all_binned_and_cats = cat_features + ['tenure_binned', 'MonthlyCharges_binned', 'TotalCharges_binned']
    for col in ['tenure_binned', 'MonthlyCharges_binned', 'TotalCharges_binned']:
        df_all[col] = df_all[col].fillna("Missing").astype(str)
        
    # Generate 171 Pairwise String combinations
    print(f"Generating 171 PAIR combinations for Logit Model...")
    pair_cols = []
    
    # Use string casting strictly to build the intersecion
    for col1, col2 in itertools.combinations(all_binned_and_cats, 2):
        pair_name = f"PAIR_{col1}_x_{col2}"
        df_all[pair_name] = df_all[col1].astype(str) + "_X_" + df_all[col2].astype(str)
        pair_cols.append(pair_name)
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    raw_cat_cols = cat_features 
    
    return X_train, y, X_test, ids, pair_cols, raw_cat_cols, base_features

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, pair_cols, raw_cat_cols, base_features = prepare_chris_deotte_features(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  V20 CHRIS DEOTTE'S 3x GPU ENSEMBLE ({NUM_FOLDS}-Fold CV)")
    print("="*60)
    
    cat_params = {
        'iterations': 2500, 'learning_rate': 0.03, 'depth': 6, 'eval_metric': 'AUC',
        'random_seed': 42, 'verbose': False, 'task_type': 'GPU'
    }
    
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 'tree_method': 'hist',
        'enable_categorical': True, 'device': 'cuda', 'learning_rate': 0.03, 'max_depth': 6,
        'random_state': 42
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_cat = np.zeros(len(X_train))
    oof_xgb = np.zeros(len(X_train))
    oof_log = np.zeros(len(X_train))
    
    test_cat = np.zeros(len(X_test))
    test_xgb = np.zeros(len(X_test))
    test_log = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr = X_train.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_va = X_train.iloc[val_idx].copy()
        y_va = y[val_idx]
        X_te = X_test.copy()
        
        # -------------------------------------------------------------
        # 1. CatBoost Model (Raw Base Features)
        # -------------------------------------------------------------
        cat_tr_pool = Pool(X_tr[base_features], y_tr, cat_features=raw_cat_cols)
        cat_va_pool = Pool(X_va[base_features], y_va, cat_features=raw_cat_cols)
        cat_te_pool = Pool(X_te[base_features], cat_features=raw_cat_cols)
        
        m_cat = CatBoostClassifier(**cat_params)
        m_cat.fit(cat_tr_pool, eval_set=cat_va_pool, early_stopping_rounds=150, verbose=False)
        
        oof_cat[val_idx] = m_cat.predict_proba(cat_va_pool)[:, 1]
        test_cat += m_cat.predict_proba(cat_te_pool)[:, 1] / NUM_FOLDS
        print(f"  [CatBoost] OOF AUC: {roc_auc_score(y_va, oof_cat[val_idx]):.5f}")
        
        # -------------------------------------------------------------
        # 2. XGBoost Model (Raw Base Features)
        # -------------------------------------------------------------
        try:
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2500, early_stopping_rounds=150)
            m_xgb.fit(X_tr[base_features], y_tr, eval_set=[(X_va[base_features], y_va)], verbose=False)
            oof_xgb[val_idx] = m_xgb.predict_proba(X_va[base_features])[:, 1]
            test_xgb += m_xgb.predict_proba(X_te[base_features])[:, 1] / NUM_FOLDS
            print(f"  [XGBoost ] OOF AUC: {roc_auc_score(y_va, oof_xgb[val_idx]):.5f}")
        except Exception as e:
            # Fallback if CUDA/Hist is not working on the user's specific setup, fallback to CPU
            print("  [XGBoost] GPU failed, falling back to CPU...")
            fallback_params = xgb_params.copy()
            fallback_params['device'] = 'cpu'
            m_xgb = xgb.XGBClassifier(**fallback_params, n_estimators=2000, early_stopping_rounds=150)
            m_xgb.fit(X_tr[base_features], y_tr, eval_set=[(X_va[base_features], y_va)], verbose=False)
            oof_xgb[val_idx] = m_xgb.predict_proba(X_va[base_features])[:, 1]
            test_xgb += m_xgb.predict_proba(X_te[base_features])[:, 1] / NUM_FOLDS
            print(f"  [XGBoost ] OOF AUC: {roc_auc_score(y_va, oof_xgb[val_idx]):.5f}")

        # -------------------------------------------------------------
        # 3. Logistic Regression (PAIR TE Features ONLY)
        # -------------------------------------------------------------
        te = TargetEncoder(cols=pair_cols, smoothing=20.0)
        
        X_tr_pairs = te.fit_transform(X_tr[pair_cols], y_tr)
        X_va_pairs = te.transform(X_va[pair_cols])
        X_te_pairs = te.transform(X_te[pair_cols])
        
        # Scale for Logistic Regression convergence
        scaler = StandardScaler()
        X_tr_pairs_sc = scaler.fit_transform(X_tr_pairs)
        X_va_pairs_sc = scaler.transform(X_va_pairs)
        X_te_pairs_sc = scaler.transform(X_te_pairs)
        
        # C=0.1 prevents TE overfitting
        m_log = LogisticRegression(C=0.1, max_iter=1000, random_state=42, solver='lbfgs')
        m_log.fit(X_tr_pairs_sc, y_tr)
        
        oof_log[val_idx] = m_log.predict_proba(X_va_pairs_sc)[:, 1]
        test_log += m_log.predict_proba(X_te_pairs_sc)[:, 1] / NUM_FOLDS
        print(f"  [Logit PAIR] OOF AUC: {roc_auc_score(y_va, oof_log[val_idx]):.5f}")

    print("\n" + "="*60)
    print("  PHASE 2: HILL CLIMBING ENSEMBLE OPTIMIZATION")
    print("="*60)
    
    print(f"Individual Global OOF AUCs:")
    print(f"  CatBoost : {roc_auc_score(y, oof_cat):.5f}")
    print(f"  XGBoost  : {roc_auc_score(y, oof_xgb):.5f}")
    print(f"  Logit TE : {roc_auc_score(y, oof_log):.5f}")

    def f_roc_auc(weights):
        # We want to MAXIMIZE AUC, scipy minimize demands a negative value
        w1, w2, w3 = weights
        blend = w1 * oof_cat + w2 * oof_xgb + w3 * oof_log
        return -roc_auc_score(y, blend)

    # Initial equal weights
    init_weights = [1/3, 1/3, 1/3]
    bounds = [(0, 1), (0, 1), (0, 1)] # Weights should generally be positive
    
    res = minimize(f_roc_auc, init_weights, bounds=bounds, method='Nelder-Mead')
    
    # Normalize weights so they sum to 1
    best_weights = res.x / np.sum(res.x)
    
    final_oof = best_weights[0] * oof_cat + best_weights[1] * oof_xgb + best_weights[2] * oof_log
    final_auc = roc_auc_score(y, final_oof)
    
    print("\n" + "="*80)
    print(f"  Final V20 (3x DEOTTE ENSEMBLE) OOF AUC: {final_auc:.5f} !!!")
    print("="*80)
    print(f"  Optimized Weights: Cat: {best_weights[0]:.3f} | XGB: {best_weights[1]:.3f} | Logit: {best_weights[2]:.3f}")
    
    if final_auc > 0.91757:
        print("\n  >> HISTORIC SUCCESS: CEILING OFFICIALLY BROKEN <<")
        
    final_test_preds = best_weights[0] * test_cat + best_weights[1] * test_xgb + best_weights[2] * test_log
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': final_test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v20_deotte_ensemble.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    with open(os.path.join(config.BASE_DIR, 'deotte_v20_report.md'), 'w') as f:
        f.write(f"# V20 Chris Deotte 3x Ensemble\n\n## Final Performance\n- **Optimized OOF AUC**: **{final_auc:.5f}**\n\n## Ensemble Weights\n- CatBoost: `{best_weights[0]:.3f}`\n- XGBoost: `{best_weights[1]:.3f}`\n- Logit PAIR TE: `{best_weights[2]:.3f}`\n")
    
    print("\nDONE! Ensembled Submission saved to:", sub_prob_path)

if __name__ == "__main__":
    main()
