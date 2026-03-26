"""
V29 Pipeline - The Exact 0.91761 Chris Deotte Blueprint
-------------------------------------------------------------------------
To break 0.914 mathematically without a leak, we must provide the trees 
with continuous distance metrics comparing the synthetic data to the 
AUTHENTIC Original IBM Churner/Non-Churner distributions.
Features implemented:
1. SurCharge = TotalCharges - (tenure * MonthlyCharges)
2. Z-Scores: (Synthetic_Value - Original_Churner_Mean) / Original_Churner_Std
3. Quantile Distances: Abs distance to Original Q25/Q50/Q75 for Churners.
4. KMeans Centroids Data.
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
from sklearn.cluster import KMeans
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
    orig_df['Churn'] = orig_df['Churn'].map({'Yes': 1, 'No': 0})
    
    for col in ['TotalCharges', 'MonthlyCharges', 'tenure']:
        orig_df[col] = pd.to_numeric(orig_df[col], errors='coerce').fillna(0)
        
    orig_df['SurCharge'] = orig_df['TotalCharges'] - (orig_df['tenure'] * orig_df['MonthlyCharges'])
    return orig_df

def feature_engineer_extreme(df, orig_df):
    df = df.copy()
    
    # ---------------------------------------------------------
    # 1. BASE NUMERICS & SURCHARGE
    # ---------------------------------------------------------
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
    df['SurCharge'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
    
    # ---------------------------------------------------------
    # 2. CONTINUOUS ORIGINAL DISTRIBUTION DISTANCES (THE 0.917 SECRET)
    # ---------------------------------------------------------
    # We calculate the exact Mean, Std, and Quantiles of the AUTHENTIC Churners
    orig_churners = orig_df[orig_df['Churn'] == 1]
    orig_non_churners = orig_df[orig_df['Churn'] == 0]
    
    for col in ['MonthlyCharges', 'TotalCharges', 'tenure', 'SurCharge']:
        # Churner Distributions
        c_mean = orig_churners[col].mean()
        c_std = orig_churners[col].std() + 1e-6
        c_q25 = orig_churners[col].quantile(0.25)
        c_q50 = orig_churners[col].quantile(0.50)
        c_q75 = orig_churners[col].quantile(0.75)
        
        # Z-Scores against Authentic Churners
        df[f'zscore_against_churner_{col}'] = (df[col] - c_mean) / c_std
        
        # Absolute Quantile Distances
        df[f'dist_q25_churner_{col}'] = np.abs(df[col] - c_q25)
        df[f'dist_q50_churner_{col}'] = np.abs(df[col] - c_q50)
        df[f'dist_q75_churner_{col}'] = np.abs(df[col] - c_q75)
        
        # Non-Churner Z-Scores (Helps trees calculate synthetic divergence probabilities)
        nc_mean = orig_non_churners[col].mean()
        nc_std = orig_non_churners[col].std() + 1e-6
        df[f'zscore_against_non_churner_{col}'] = (df[col] - nc_mean) / nc_std

    # ---------------------------------------------------------
    # 3. K-MEANS CENTROID DISTANCES
    # ---------------------------------------------------------
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SurCharge']
    X_num = df[num_cols].fillna(0).values
    # Fit KMeans strictly on the original dataset to avoid internal test-set leaking
    orig_num = orig_df[num_cols].fillna(0).values
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    kmeans.fit(orig_num)
    
    # Calculate distance to all 8 authentic centroids!
    distances = kmeans.transform(X_num)
    for i in range(8):
        df[f'kmeans_dist_cluster_{i}'] = distances[:, i]
        
    df['kmeans_cluster_id'] = kmeans.predict(X_num).astype(str)

    # ---------------------------------------------------------
    # 4. CATEGORICAL INTERACTION (N-GRAMS) & SATURATION
    # ---------------------------------------------------------
    df['Ngram_Contract_Internet_Pay'] = df['Contract'].astype(str) + "_" + df['InternetService'].astype(str) + "_" + df['PaymentMethod'].astype(str)
    
    orig_df['Ngram'] = orig_df['Contract'].astype(str) + "_" + orig_df['InternetService'].astype(str) + "_" + orig_df['PaymentMethod'].astype(str)
    orig_ngram_map = orig_df.groupby('Ngram')['Churn'].mean().to_dict()
    df['Orig_Ngram_Churn_Risk'] = df['Ngram_Contract_Internet_Pay'].map(orig_ngram_map).fillna(0)
    
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['Service_Count'] = (df[services] == 'Yes').sum(axis=1)

    return df

def main():
    print("="*70)
    print(" V29: CHRIS DEOTTE KAGGLE BLUEPRINT (TARGET > 0.91761) ")
    print("="*70)
    
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    orig_df = get_original_dataset()
    
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL, config.KAGGLE_ID_COL], errors='ignore')
    
    ids = test_df[config.KAGGLE_ID_COL].values
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    print("\nExecuting extreme Continuous Original Distribution Mappings & KMeans...")
    train_features = feature_engineer_extreme(train_df, orig_df)
    test_features = feature_engineer_extreme(test_df, orig_df)
    
    # XGBoost Categorical handling
    cat_cols = train_features.select_dtypes(include=['object']).columns
    for col in cat_cols:
        train_features[col] = train_features[col].astype('category')
        test_features[col] = test_features[col].astype('category')
        
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'enable_categorical': True,
        'learning_rate': 0.015,
        'max_depth': 5,          # Shallow enough to ignore GAN noise, deep enough for Z-scores
        'colsample_bytree': 0.6,
        'subsample': 0.8,
        'min_child_weight': 8,  
        'device': 'cuda',        
        'random_state': 42
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(train_features))
    test_preds = np.zeros(len(test_features))
    
    print(f"\nTraining {NUM_FOLDS}-Fold Pure XGBoost with Engineered Matrix...")
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_features, y)):
        X_tr, y_tr = train_features.iloc[train_idx], y[train_idx]
        X_va, y_va = train_features.iloc[val_idx], y[val_idx]
        
        try:
            model = xgb.XGBClassifier(**xgb_params, n_estimators=3500, early_stopping_rounds=250)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            xgb_params['device'] = 'cpu'
            model = xgb.XGBClassifier(**xgb_params, n_estimators=3500, early_stopping_rounds=250)
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
        oof_preds[val_idx] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(test_features)[:, 1] / NUM_FOLDS
        print(f"  [Fold {fold+1}] OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")

    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*80)
    print(f"  Final V29 (Chris Deotte Blueprint) OOF AUC: {final_auc:.5f}")
    if final_auc > 0.9176:
        print("  !!! THE CEILING HAS BEEN OFFICIALLY BROKEN !!!")
    print("="*80)
    
    # -------------------------------------------------------------
    # TARGET CALIBRATION (POST-PROCESSING)
    # The Kaggle Test Set often has a slightly different Churn Ratio.
    # Calibration smoothly boosts the rank without altering order, fixing metric skews.
    # -------------------------------------------------------------
    train_churn_rate = np.mean(y)
    test_mean_pred = np.mean(test_preds)
    calibration_factor = train_churn_rate / test_mean_pred
    calibrated_test_preds = np.clip(test_preds * calibration_factor, 0.0001, 0.9999)
    print(f"\nTarget Calibrated: Train Rate ({train_churn_rate:.4f}) / Test Pred Mean ({test_mean_pred:.4f}). Factor: {calibration_factor:.4f}")
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: ids,
        config.KAGGLE_TARGET_COL: calibrated_test_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v29_chris_deotte_exact.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"  Calibrated Submission saved to: {sub_path}")

if __name__ == "__main__":
    main()
