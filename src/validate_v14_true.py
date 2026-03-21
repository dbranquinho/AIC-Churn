"""
Validates the V14 Pseudo-Labeling model by calculating the Out-Of-Fold (OOF) AUC exclusively on the original training data.
The inflated 0.92968 AUC comes from evaluating against the highly confident pseudo-labels.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')

# V14 PSEUDO-LABELING PARAMETERS
PSEUDO_POS_THRESHOLD = 0.985
PSEUDO_NEG_THRESHOLD = 0.015
NUM_FOLDS = 10

def feature_engineering(train_df, test_df):
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    original_train_len = len(train_df)
    
    # --- V14 PSEUDO LABELING INJECTION ---
    v13_preds_path = os.path.join(SUBMISSION_DIR, 'submission_v13_target_encoded.csv')
    
    if os.path.exists(v13_preds_path):
        v13_df = pd.read_csv(v13_preds_path)
        pos_mask = v13_df['Churn'] >= PSEUDO_POS_THRESHOLD
        neg_mask = v13_df['Churn'] <= PSEUDO_NEG_THRESHOLD
        
        test_pseudo_pos = test_df[pos_mask].copy()
        test_pseudo_neg = test_df[neg_mask].copy()
        
        pseudo_train = pd.concat([test_pseudo_pos, test_pseudo_neg], axis=0)
        pseudo_labels = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
        
        if len(pseudo_train) > 0:
            train_df = pd.concat([train_df, pseudo_train], axis=0).reset_index(drop=True)
            y = np.concatenate([y, pseudo_labels])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    cat_cols = config.KAGGLE_CATEGORICAL_COLS.copy()
    
    # Base Combo
    df_all['Contract_Payment'] = df_all['Contract'] + "_" + df_all['PaymentMethod']
    cat_cols.append('Contract_Payment')

    # Revert to fillna(0) for stability
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
    # ChargeResidual Trick
    expected_charge = df_all['MonthlyCharges'] * df_all['tenure']
    charge_residual = df_all['TotalCharges'] - expected_charge
    
    df_all['charge_residual'] = charge_residual
    df_all['charge_residual_sign'] = np.sign(charge_residual)
    df_all['charge_residual_relative'] = charge_residual / (expected_charge + 1)
    
    # Logical Ratios
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + 1)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + 1)
    
    for col in cat_cols:
        df_all[col] = df_all[col].fillna('Missing').astype(str)
        
    X_train = df_all.iloc[:train_len].copy()
    
    return X_train, y, cat_cols, original_train_len


def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, cat_cols, original_train_len = feature_engineering(train_df, test_df)
    
    best_xgb_p = {
      "learning_rate": 0.03229243963427607,
      "max_depth": 6,
      "subsample": 0.8501376696196967,
      "colsample_bytree": 0.3016242963770833,
      "min_child_weight": 18,
      "gamma": 0.003627484233742829,
      "reg_alpha": 0.1135966159982872,
      "reg_lambda": 12.095855865207273,
      "objective": "binary:logistic",
      "eval_metric": "auc",
      "tree_method": "hist",
      "device": "cuda",
      "random_state": 42
    }
    best_te_smoothing = 37.19463659026127
    
    print("\n" + "="*60)
    print(f"  Validating Final {NUM_FOLDS}-Fold Target Encoded Pseudo-XGBoost")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    # We only care about True OOF (original train size)
    oof_true = np.zeros(original_train_len)
    y_true = y[:original_train_len]
    
    inflated_oof = np.zeros(len(X_train))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        te = TargetEncoder(cols=cat_cols, smoothing=best_te_smoothing)
        X_tr_enc = te.fit_transform(X_tr, y_tr)
        X_va_enc = te.transform(X_va)
        
        dtrain_xgb = xgb.DMatrix(X_tr_enc, label=y_tr)
        dvalid_xgb = xgb.DMatrix(X_va_enc, label=y_va)
        
        model_xgb = xgb.train(best_xgb_p, dtrain_xgb, 
                              num_boost_round=3500, 
                              evals=[(dvalid_xgb, 'valid')], 
                              early_stopping_rounds=200, 
                              verbose_eval=False)
                              
        preds = model_xgb.predict(dvalid_xgb)
        inflated_oof[val_idx] = preds
        
        # Filter true validation samples
        true_val_mask = val_idx < original_train_len
        if true_val_mask.any():
            true_val_idx = val_idx[true_val_mask]
            true_preds = preds[true_val_mask]
            oof_true[true_val_idx] = true_preds
        
    inflated_auc = roc_auc_score(y, inflated_oof)
    true_auc = roc_auc_score(y_true, oof_true)
    
    print("\n" + "="*60)
    print(f"  Inflated V14 OOF AUC (Train + Pseudo): {inflated_auc:.5f} (Matches kaggle_train.py metric)")
    print(f"  TRUE V14 OOF AUC (Train Only):        {true_auc:.5f}")
    print("="*60)
    print("The inflated score occurs because predicting the pseudo-labels is highly biased.")

if __name__ == "__main__":
    main()
