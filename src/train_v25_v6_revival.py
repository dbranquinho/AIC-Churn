"""
V25 Pipeline - The V6 Pure Tuning Revival (Maximum Actual LB Score)
-------------------------------------------------------------------------
History has proven that the V6 logic ("Zero Feature Engineering" + "Tri-Ensemble")
is the only architecture that cleanly surpasses the 0.914 LB wall.
This script resurrects the precise V6 philosophy:
1. Pure Raw Features ONLY (Letting Trees handle NaNs natively).
2. XGBoost + LightGBM + CatBoost trained concurrently with robust tuning.
3. Scipy Hill Climbing Optimization rather than naive simple averaging, 
   supercharging the old V6 ensemble mechanism.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')
import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
NUM_FOLDS = 10

def get_raw_v6_data():
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL, config.KAGGLE_ID_COL], errors='ignore')
    
    ids = test_df[config.KAGGLE_ID_COL].values
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce')
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce')
        
    cat_cols = [c for c in train_df.columns if c not in numeric_cols]
    
    for col in cat_cols:
        # V6 clean formatting
        train_df[col] = train_df[col].fillna('Missing').astype(str).astype('category')
        test_df[col] = test_df[col].fillna('Missing').astype(str).astype('category')
        
    return train_df, y, test_df, ids, cat_cols

def f_roc_auc(weights, oof_xgb, oof_lgb, oof_cat, y):
    blend = weights[0] * oof_xgb + weights[1] * oof_lgb + weights[2] * oof_cat
    return -roc_auc_score(y, blend)

def main():
    print("="*70)
    print(" V25: V6 PURE TUNED TRI-ENSEMBLE REVIVAL ")
    print("="*70)
    
    X_train, y, X_test, test_ids, cat_cols = get_raw_v6_data()
    
    # ---------------------------------------------------------
    # SUPER-TUNED ROBUST PARAMETERS (No Optuna Wait Needed)
    # ---------------------------------------------------------
    xgb_params = {
        'objective': 'binary:logistic', 'eval_metric': 'auc', 
        'tree_method': 'hist', 'enable_categorical': True, 'device': 'cuda',
        'learning_rate': 0.015, 'max_depth': 5, 'colsample_bytree': 0.7,
        'subsample': 0.8, 'min_child_weight': 5, 'random_state': 42
    }
    
    lgb_params = {
        'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.015,
        'max_depth': 5, 'num_leaves': 15, 'min_child_samples': 20,
        'colsample_bytree': 0.7, 'subsample': 0.8, 'reg_alpha': 1.0,
        'reg_lambda': 3.0, 'random_state': 42, 'verbose': -1
    }
    
    cat_params = {
        'eval_metric': 'AUC', 'learning_rate': 0.015, 'depth': 6,
        'l2_leaf_reg': 3.0, 'random_seed': 42, 'task_type': 'GPU', 'verbose': False
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))
    
    test_xgb = np.zeros(len(X_test))
    test_lgb = np.zeros(len(X_test))
    test_cat = np.zeros(len(X_test))
    
    print("\nStarting Parallel Native Training (No Magic Features)...\n")
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[t_idx], y[t_idx]
        X_va, y_va = X_train.iloc[v_idx], y[v_idx]
        
        # 1. XGBoost GPU
        try:
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2500, early_stopping_rounds=200)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            xgb_params['device'] = 'cpu'
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2500, early_stopping_rounds=200)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
        oof_xgb[v_idx] = m_xgb.predict_proba(X_va)[:, 1]
        test_xgb += m_xgb.predict_proba(X_test)[:, 1] / NUM_FOLDS
        
        # 2. LightGBM (Native Category Handle)
        m_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=2500)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
        oof_lgb[v_idx] = m_lgb.predict_proba(X_va)[:, 1]
        test_lgb += m_lgb.predict_proba(X_test)[:, 1] / NUM_FOLDS
        
        # 3. CatBoost GPU (Native String Cat Handle needed, so we pass python list of strings)
        cat_str_cols = list(cat_cols)
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_str_cols)
        va_pool = Pool(X_va, y_va, cat_features=cat_str_cols)
        te_pool = Pool(X_test, cat_features=cat_str_cols)
        
        m_cat = CatBoostClassifier(**cat_params, iterations=2500)
        m_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=200, verbose=False)
        
        oof_cat[v_idx] = m_cat.predict_proba(va_pool)[:, 1]
        test_cat += m_cat.predict_proba(te_pool)[:, 1] / NUM_FOLDS
        
        print(f"--- FOLD {fold+1} --- XGB: {roc_auc_score(y_va, oof_xgb[v_idx]):.5f} | LGB: {roc_auc_score(y_va, oof_lgb[v_idx]):.5f} | CAT: {roc_auc_score(y_va, oof_cat[v_idx]):.5f}")

    print("\n" + "="*70)
    print("  HILL CLIMBING ENSEMBLE OPTIMIZATION (V20 Method on V6 Logic) ")
    print("="*70)
    
    res = minimize(f_roc_auc, [1/3, 1/3, 1/3], args=(oof_xgb, oof_lgb, oof_cat, y), bounds=[(0, 1)]*3, method='Nelder-Mead')
    weights = res.x / np.sum(res.x)
    
    final_oof = weights[0] * oof_xgb + weights[1] * oof_lgb + weights[2] * oof_cat
    final_auc = roc_auc_score(y, final_oof)
    
    print(f"\nOptimization Finished! Final Blended OOF AUC: {final_auc:.5f}")
    print(f"Weights Found -> XGBoost: {weights[0]:.3f} | LightGBM: {weights[1]:.3f} | CatBoost: {weights[2]:.3f}")
    
    final_test_preds = weights[0] * test_xgb + weights[1] * test_lgb + weights[2] * test_cat
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: test_ids,
        config.KAGGLE_TARGET_COL: final_test_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v25_v6_revival.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"\nFinal V25 (V6 Revival) saved to: {sub_path}")

if __name__ == "__main__":
    main()
