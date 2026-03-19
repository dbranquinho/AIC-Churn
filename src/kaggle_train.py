"""
Kaggle Playground Series S6E3 — V7 Optimized Ensemble
Integrates pure native categorical parsing, robust continuous financial features,
and mathematically optimized blending weights to maximize ROC-AUC.
"""
import os
import numpy as np
import pandas as pd
import optuna
import warnings
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

import xgboost as xgb
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

warnings.filterwarnings('ignore')
import src.config as config

KAGGLE_MODEL_DIR = os.path.join(config.MODEL_DIR, 'kaggle')
os.makedirs(KAGGLE_MODEL_DIR, exist_ok=True)
SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

NUM_FOLDS = 10
N_TRIALS = 30  # Deep evaluation: 30 Optuna trials per model. 

def feature_engineering(train_df, test_df):
    """V6 Clean: Unaltered core features to eliminate synthetic cross-math leakage."""
    print("Applying deeply cleaned feature engineering (v6)...")
    
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0}).values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    # 1. Clean Categorical Identification
    cat_cols = config.KAGGLE_CATEGORICAL_COLS.copy()
    
    # Optional logic: simple additive crosses that don't invent numbers
    df_all['Contract_Payment'] = df_all['Contract'] + "_" + df_all['PaymentMethod']
    cat_cols.append('Contract_Payment')

    # 2. Fix empty numerics
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    
    # V7 Financial Features
    df_all['avg_charge_per_tenure'] = df_all['TotalCharges'] / (df_all['tenure'] + 1)
    df_all['charge_discrepancy'] = df_all['TotalCharges'] - (df_all['MonthlyCharges'] * df_all['tenure'])
    df_all['pct_discrepancy'] = df_all['charge_discrepancy'] / (df_all['TotalCharges'] + 1)
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + 1)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + 1)
    
    # 3. Categorical Processing - pure native casting
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        df_all[col] = df_all[col].astype('category')
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    print(f"Final Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y, X_test, ids, cat_cols


# ================= OPUNTA OBJECTIVES =================

def obj_xgb(trial, X, y):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'enable_categorical': True,
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42
    }
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    
    model = xgb.train(params, dtrain, num_boost_round=1500, evals=[(dvalid, 'valid')], 
                      early_stopping_rounds=100, verbose_eval=False)
    return model.best_score

def obj_lgb(trial, X, y):
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 15, 127),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.5, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1
    }
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    model = LGBMClassifier(**params, n_estimators=1000)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    
    # Some older LGBM versions evaluate string arrays differently
    return roc_auc_score(y_va, model.predict_proba(X_va)[:, 1])

def obj_cat(trial, X, y, cat_cols):
    params = {
        'iterations': 1000,
        'eval_metric': 'AUC',
        'task_type': 'GPU',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.08, log=True),
        'depth': trial.suggest_int('depth', 4, 8),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 20, log=True),
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)
    
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool)
    return model.get_best_score()['validation']['AUC']

# ======================================================

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, cat_cols = feature_engineering(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  Phase 1: Deep Tuning Across 3 Models ({N_TRIALS} Trials Each)")
    print("="*60)
    
    # Pruners save incredible amounts of time safely
    pruner_cb = optuna.pruners.MedianPruner(n_warmup_steps=10)
    
    print(f"\n[1/3] Tuning XGBoost...")
    study_xgb = optuna.create_study(direction="maximize", pruner=pruner_cb)
    study_xgb.optimize(lambda t: obj_xgb(t, X_train, y), n_trials=N_TRIALS, n_jobs=1)
    best_xgb_p = study_xgb.best_trial.params
    best_xgb_p.update({'objective': 'binary:logistic', 'eval_metric': 'auc', 
                       'tree_method': 'hist', 'device': 'cuda', 'enable_categorical': True, 'random_state': 42})
    print(f"Best XGB AUC: {study_xgb.best_value:.5f}")
    
    print(f"\n[2/3] Tuning LightGBM...")
    study_lgb = optuna.create_study(direction="maximize", pruner=pruner_cb)
    study_lgb.optimize(lambda t: obj_lgb(t, X_train, y), n_trials=N_TRIALS, n_jobs=1)
    best_lgb_p = study_lgb.best_trial.params
    best_lgb_p.update({'objective': 'binary', 'metric': 'auc', 'random_state': 42, 'verbose': -1, 'n_jobs': -1})
    print(f"Best LGB AUC: {study_lgb.best_value:.5f}")
    
    print(f"\n[3/3] Tuning CatBoost...")
    study_cat = optuna.create_study(direction="maximize", pruner=pruner_cb)
    study_cat.optimize(lambda t: obj_cat(t, X_train, y, cat_cols), n_trials=N_TRIALS, n_jobs=1)
    best_cat_p = study_cat.best_trial.params
    best_cat_p.update({'iterations': 2000, 'eval_metric': 'AUC', 'task_type': 'GPU', 
                       'random_seed': 42, 'verbose': False, 'early_stopping_rounds': 100})
    print(f"Best CAT AUC: {study_cat.best_value:.5f}")
    
    print("\n" + "="*60)
    print(f"  Phase 2: Final 10-Fold Ensembling with Tuned Params")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(X_train))
    oof_lgb = np.zeros(len(X_train))
    oof_cat = np.zeros(len(X_train))
    preds_xgb = np.zeros(len(X_test))
    preds_lgb = np.zeros(len(X_test))
    preds_cat = np.zeros(len(X_test))
    
    dtest_xgb = xgb.DMatrix(X_test, enable_categorical=True)
    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]
    test_pool = Pool(X_test, cat_features=cat_idx)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        # XGB
        dtrain_xgb = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid_xgb = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        model_xgb = xgb.train(best_xgb_p, dtrain_xgb, num_boost_round=2500, evals=[(dvalid_xgb, 'valid')], early_stopping_rounds=150, verbose_eval=False)
        oof_xgb[val_idx] = model_xgb.predict(dvalid_xgb)
        preds_xgb += model_xgb.predict(dtest_xgb) / NUM_FOLDS
        
        # LGB
        model_lgb = LGBMClassifier(**best_lgb_p, n_estimators=2000)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        oof_lgb[val_idx] = model_lgb.predict_proba(X_va)[:, 1]
        preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / NUM_FOLDS
        
        # CAT
        train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
        valid_pool = Pool(X_va, y_va, cat_features=cat_idx)
        model_cat = CatBoostClassifier(**best_cat_p)
        model_cat.fit(train_pool, eval_set=valid_pool)
        oof_cat[val_idx] = model_cat.predict_proba(valid_pool)[:, 1]
        preds_cat += model_cat.predict_proba(test_pool)[:, 1] / NUM_FOLDS
        
    from scipy.optimize import minimize
    print("\n" + "="*60)
    print(f"  Tuned XGBoost  AUC : {roc_auc_score(y, oof_xgb):.5f}")
    print(f"  Tuned LightGBM AUC : {roc_auc_score(y, oof_lgb):.5f}")
    print(f"  Tuned CatBoost AUC : {roc_auc_score(y, oof_cat):.5f}")
    
    print("\n  Optimizing Ensemble Weights...")
    def obj_func(weights):
        w_xgb, w_lgb, w_cat = weights
        blend = w_xgb * oof_xgb + w_lgb * oof_lgb + w_cat * oof_cat
        return -roc_auc_score(y, blend)
        
    res = minimize(obj_func, [1/3, 1/3, 1/3], method='Nelder-Mead')
    best_weights = res.x
    best_weights /= np.sum(best_weights) # normalize
    
    oof_ensemble = best_weights[0]*oof_xgb + best_weights[1]*oof_lgb + best_weights[2]*oof_cat
    final_auc = roc_auc_score(y, oof_ensemble)
    print(f"\n  Final V7 Optimized Ensemble OOF AUC: {final_auc:.5f} !!!")
    print(f"  Weights -> XGB: {best_weights[0]:.4f}, LGB: {best_weights[1]:.4f}, CAT: {best_weights[2]:.4f}")
    print("="*60)
    
    final_test_preds = best_weights[0]*preds_xgb + best_weights[1]*preds_lgb + best_weights[2]*preds_cat
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': final_test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v7_optimized.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    report_content = f"""# V7 Optimized Ensemble Report

## Strategy
1. **Financial Features**: Re-added avg_charge_per_tenure, discrepancies, cross ratios based on v3 success.
2. **Deep Tuning**: Dedicated 30-trial Optuna study matching hyperparameters specifically for each model independently.
3. **Optimized Ensembling**: Utilized Nelder-Mead optimization to find the exact blend weights `(XGB, LGB, CAT)`.
    
## Final Performance
- **Tuned XGBoost AUC**: {roc_auc_score(y, oof_xgb):.5f}
- **Tuned LightGBM AUC**: {roc_auc_score(y, oof_lgb):.5f}
- **Tuned CatBoost AUC**: {roc_auc_score(y, oof_cat):.5f}
- **Optimized Blend Weights**: XGB: {best_weights[0]:.4f}, LGB: {best_weights[1]:.4f}, CAT: {best_weights[2]:.4f}
- **Final Optimized Ensemble OOF AUC**: **{final_auc:.5f}**
"""
    with open(os.path.join(config.BASE_DIR, 'xgb_v7_report.md'), 'w') as f:
        f.write(report_content)
    print("\nDONE! Report saved to xgb_v7_report.md.")

if __name__ == "__main__":
    main()
