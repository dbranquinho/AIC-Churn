"""
V24 Pipeline - The Kaggle "Magic Features" 0.91761+ Strategy
-------------------------------------------------------------------------
Based on the top active Kaggle S6E3 solutions:
1. "SurCharge" (Hidden Fees/GAN error proxy) = TotalCharges - (tenure * MonthlyCharges)
2. Original Target Coding: V11 failed because concatenating IBM Dataset rows
   ruins the CTGAN split ratios. The secret is to use the IBM dataset ONLY 
   as a purely statistical mapping dictionary to create continuous features!
3. N-Grams: Contract_Internet_Payment interactions.
4. Total Service Count.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import urllib.request
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings

warnings.filterwarnings('ignore')
import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
NUM_FOLDS = 10

def get_original_dataset():
    orig_path = os.path.join(config.BASE_DIR, 'data', 'true_original_telco.csv')
    if not os.path.exists(orig_path):
        url = 'https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv'
        print("  [Download] Fetching Original IBM Telco Dataset as Statistical Dictionary...")
        urllib.request.urlretrieve(url, orig_path)
    
    orig_df = pd.read_csv(orig_path)
    
    # Clean map format to match Kaggle
    orig_df['Churn'] = orig_df['Churn'].map({'Yes': 1, 'No': 0})
    orig_df['TotalCharges'] = pd.to_numeric(orig_df['TotalCharges'], errors='coerce')
    return orig_df

def feature_engineer(df, orig_df=None):
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. MAGIC FEATURE: SurCharge (The GAN / Fee discrepancy)
    # ---------------------------------------------------------
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    
    df['SurCharge'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
    
    # ---------------------------------------------------------
    # 2. MAGIC CATEGORICAL N-GRAM
    # ---------------------------------------------------------
    df['Magic_Ngram'] = df['Contract'].astype(str) + "_" + df['InternetService'].astype(str) + "_" + df['PaymentMethod'].astype(str)
    
    # ---------------------------------------------------------
    # 3. SERVICE SATURATION
    # ---------------------------------------------------------
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = (df[services] == 'Yes').sum(axis=1)
    
    # ---------------------------------------------------------
    # 4. ORIGINAL DATASET STATISTICAL DICTIONARY (NO CONCATENATION!)
    # ---------------------------------------------------------
    if orig_df is not None:
        # We calculate the precise historic churn rate of the Categorical Ngram from the *real* 1990s dataset
        orig_df['Magic_Ngram'] = orig_df['Contract'].astype(str) + "_" + orig_df['InternetService'].astype(str) + "_" + orig_df['PaymentMethod'].astype(str)
        
        orig_ngram_map = orig_df.groupby('Magic_Ngram')['Churn'].mean().to_dict()
        df['Orig_Ngram_Churn_Risk'] = df['Magic_Ngram'].map(orig_ngram_map).fillna(0)
        
        orig_contract_map = orig_df.groupby('Contract')['Churn'].mean().to_dict()
        df['Orig_Contract_Risk'] = df['Contract'].map(orig_contract_map).fillna(0)
        
    return df

def main():
    print("="*70)
    print(" V24 KAGGLE S6E3 MAGIC FEATURES (Target OOF > 0.9176) ")
    print("="*70)
    
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    orig_df = get_original_dataset()
    
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL, config.KAGGLE_ID_COL], errors='ignore')
    
    ids = test_df[config.KAGGLE_ID_COL].values
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    print("\nExtracting Magic Features and Original Historic Risks...")
    train_features = feature_engineer(train_df, orig_df)
    test_features = feature_engineer(test_df, orig_df)
    
    # Convert all object columns to categorical for XGBoost
    cat_cols = train_features.select_dtypes(include=['object']).columns
    for col in cat_cols:
        train_features[col] = train_features[col].astype('category')
        test_features[col] = test_features[col].astype('category')
        
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'enable_categorical': True,
        'learning_rate': 0.02,
        'max_depth': 5,          # Shallow enough to generalize
        'colsample_bytree': 0.7,
        'subsample': 0.8,
        'min_child_weight': 10,  # Prune noise
        'device': 'cuda',        # Force GPU
        'random_state': 42
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_features))
    test_preds = np.zeros(len(test_features))
    
    print(f"\nTraining {NUM_FOLDS}-Fold XGBoost with Auto Categorical...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y)):
        X_tr, y_tr = train_features.iloc[train_idx], y[train_idx]
        X_va, y_va = train_features.iloc[val_idx], y[val_idx]
        
        try:
            model = xgb.XGBClassifier(**xgb_params, n_estimators=3000, early_stopping_rounds=200)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            # Fallback to CPU if GPU driver fails
            xgb_params['device'] = 'cpu'
            model = xgb.XGBClassifier(**xgb_params, n_estimators=3000, early_stopping_rounds=200)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(test_features)[:, 1] / NUM_FOLDS
        print(f"  [Fold {fold+1}] OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")

    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*80)
    print(f"  Final V24 (Kaggle Magic Features) OOF AUC: {final_auc:.5f}")
    if final_auc > 0.9176:
        print("  !!! THE CEILING HAS BEEN OFFICIALLY BROKEN !!!")
    print("="*80)
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: ids,
        config.KAGGLE_TARGET_COL: test_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v24_kaggle_magic.csv')
    sub_df.to_csv(sub_path, index=False)
    
    print(f"  Submission saved to: {sub_path}")

if __name__ == "__main__":
    main()
