"""
Kaggle Playground Series S6E3 — V14 The Pseudo-Labeling Booster
Implements the ultimate Kaggle Semi-Supervised trick.
Injects highly confident Test predictions (from V13) back into the Training Set
to forcibly shift the mathematical bounds towards the Test Distribution.
"""
import os
import numpy as np
import pandas as pd
import optuna
import warnings
import json
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder

import xgboost as xgb

warnings.filterwarnings('ignore')
import src.config as config

KAGGLE_MODEL_DIR = os.path.join(config.MODEL_DIR, 'kaggle')
os.makedirs(KAGGLE_MODEL_DIR, exist_ok=True)
SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

# V14 PSEUDO-LABELING PARAMETERS
PSEUDO_POS_THRESHOLD = 0.985
PSEUDO_NEG_THRESHOLD = 0.015

NUM_FOLDS = 10
N_TRIALS = 30  # Faster tuning, relying on the injected data distributions

def feature_engineering(train_df, test_df):
    print("Applying V14 Pseudo-Labeling + Target Encoding...")
    
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    # --- V14 PSEUDO LABELING INJECTION ---
    v13_preds_path = os.path.join(SUBMISSION_DIR, 'submission_v13_target_encoded.csv')
    pseudo_y = []
    
    if os.path.exists(v13_preds_path):
        v13_df = pd.read_csv(v13_preds_path)
        # Find highly confident predictions in Test Set
        pos_mask = v13_df['Churn'] >= PSEUDO_POS_THRESHOLD
        neg_mask = v13_df['Churn'] <= PSEUDO_NEG_THRESHOLD
        
        test_pseudo_pos = test_df[pos_mask].copy()
        test_pseudo_neg = test_df[neg_mask].copy()
        
        print(f"V14: Found {pos_mask.sum()} HIGH confidence Positive Churn tests.")
        print(f"V14: Found {neg_mask.sum()} HIGH confidence Negative Churn tests.")
        
        pseudo_train = pd.concat([test_pseudo_pos, test_pseudo_neg], axis=0)
        pseudo_labels = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
        
        # Inject back to Training
        if len(pseudo_train) > 0:
            print("V14: INJECTING Test records back into Train!")
            train_df = pd.concat([train_df, pseudo_train], axis=0).reset_index(drop=True)
            y = np.concatenate([y, pseudo_labels])
    else:
        print("V14 WARNING: v13 predictions not found, pseudo-labeling skipped.")
    
    # -------------------------------------
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df) # New length includes pseudo labels
    
    cat_cols = config.KAGGLE_CATEGORICAL_COLS.copy()
    
    # 1. Base Combo
    df_all['Contract_Payment'] = df_all['Contract'] + "_" + df_all['PaymentMethod']
    cat_cols.append('Contract_Payment')

    # 2. Revert to fillna(0) for stability
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
    # 3. The Grandmaster's "ChargeResidual" Trick (V13/V14 core)
    expected_charge = df_all['MonthlyCharges'] * df_all['tenure']
    charge_residual = df_all['TotalCharges'] - expected_charge
    
    df_all['charge_residual'] = charge_residual
    df_all['charge_residual_sign'] = np.sign(charge_residual)
    df_all['charge_residual_relative'] = charge_residual / (expected_charge + 1)
    
    # 4. Standard Logical Ratios
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + 1)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + 1)
    
    # Fill categorical nulls
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    print(f"Final Train shape (Synthetic + Pseudo): {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y, X_test, ids, cat_cols


# ================= OPUNTA OBJECTIVES =================

def obj_xgb(trial, X, y, cat_cols):
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 9),
        'subsample': trial.suggest_float('subsample', 0.4, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 0.95),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
        'gamma': trial.suggest_float('gamma', 0.001, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-4, 50, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-4, 50, log=True),
        'random_state': 42
    }
    
    # Create single validation split for Optuna
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # In-Fold Target Encoding (Smoothing parameter tuned)
    te = TargetEncoder(cols=cat_cols, smoothing=trial.suggest_float('te_smoothing', 1.0, 50.0))
    X_tr_enc = te.fit_transform(X_tr, y_tr)
    X_va_enc = te.transform(X_va)
    
    dtrain = xgb.DMatrix(X_tr_enc, label=y_tr)
    dvalid = xgb.DMatrix(X_va_enc, label=y_va)
    
    model = xgb.train(params, dtrain, num_boost_round=2500, 
                      evals=[(dvalid, 'valid')], 
                      early_stopping_rounds=150, verbose_eval=False)
    
    return model.best_score

# ======================================================

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, cat_cols = feature_engineering(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  Phase 1: Deep Tuning Pseudo-XGBoost + TargetEncoder ({N_TRIALS} Trials)")
    print("="*60)
    
    pruner_cb = optuna.pruners.MedianPruner(n_warmup_steps=10)
    study_xgb = optuna.create_study(direction="maximize", pruner=pruner_cb)
    study_xgb.optimize(lambda t: obj_xgb(t, X_train, y, cat_cols), n_trials=N_TRIALS, n_jobs=1)
    
    best_xgb_p = study_xgb.best_trial.params
    best_te_smoothing = best_xgb_p.pop('te_smoothing')
    
    best_xgb_p.update({
        'objective': 'binary:logistic', 
        'eval_metric': 'auc', 
        'tree_method': 'hist', 
        'device': 'cuda', 
        'random_state': 42
    })
    print(f"Best Pseudo XGB AUC via Optuna Validation: {study_xgb.best_value:.5f}")
    
    print("\n" + "="*60)
    print(f"  Phase 2: Final {NUM_FOLDS}-Fold Target Encoded Pseudo-XGBoost")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_xgb = np.zeros(len(X_train))
    preds_xgb = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        # Strict In-Fold Target Encoding
        te = TargetEncoder(cols=cat_cols, smoothing=best_te_smoothing)
        X_tr_enc = te.fit_transform(X_tr, y_tr)
        X_va_enc = te.transform(X_va)
        X_test_enc = te.transform(X_test)
        
        dtrain_xgb = xgb.DMatrix(X_tr_enc, label=y_tr)
        dvalid_xgb = xgb.DMatrix(X_va_enc, label=y_va)
        dtest_xgb = xgb.DMatrix(X_test_enc)
        
        model_xgb = xgb.train(best_xgb_p, dtrain_xgb, 
                              num_boost_round=3500, 
                              evals=[(dvalid_xgb, 'valid')], 
                              early_stopping_rounds=200, 
                              verbose_eval=500)
                              
        oof_xgb[val_idx] = model_xgb.predict(dvalid_xgb)
        preds_xgb += model_xgb.predict(dtest_xgb) / NUM_FOLDS
        
    final_auc = roc_auc_score(y, oof_xgb)
    print("\n" + "="*60)
    print(f"  Final V14 'Pseudo-Labeling Booster' OOF AUC: {final_auc:.5f} !!!")
    print("="*60)
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': preds_xgb})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v14_pseudo_labeled.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    best_xgb_p['te_smoothing'] = best_te_smoothing
    params_str = json.dumps(best_xgb_p, indent=2)
    
    # Calculate pseudo injection stats
    v13_preds_path = os.path.join(SUBMISSION_DIR, 'submission_v13_target_encoded.csv')
    if os.path.exists(v13_preds_path):
        v13_df = pd.read_csv(v13_preds_path)
        # Use simple limits just for logging purposes here, actual matching uses bounds set via globals
        pos = (v13_df['Churn'] >= 0.985).sum()
        neg = (v13_df['Churn'] <= 0.015).sum()
        pseudo_stats = f"Injected {pos+neg} test records (Pos: {pos}, Neg: {neg})"
    else:
        pseudo_stats = "Failed to inject."
        
    report_content = f"""# V14 Pseudo-Labeling & Semi-Supervised Learning XGBoost

## Strategy
1. **The Overfitting Panacea**: Reached the mathematical limit of the training data. Shifted to Semi-Supervised Pseudo-Labeling.
2. **Confidence Injection**: Read the `submission_v13` probabilities. **{pseudo_stats}** with >98.5% and <1.5% confidence were given fake `Churn` targets and appended securely back to `train.csv`.
3. **Target Encoding**: Retained `TargetEncoder` and `ChargeResidual`.
    
## Final Performance
- **10-Fold OOF AUC (Train + Pseudo)**: **{final_auc:.5f}**

## Top Best Parameters Found
```json
{params_str}
```
"""

    with open(os.path.join(config.BASE_DIR, 'xgb_v14_report.md'), 'w') as f:
        f.write(report_content)
    print("\nDONE! Report saved to xgb_v14_report.md.")

if __name__ == "__main__":
    main()
