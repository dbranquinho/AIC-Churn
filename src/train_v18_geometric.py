"""
V18 Pipeline - Deep Research: Unsupervised Geometric Manifolds
Since Adversarial Validation proved that dropping drifting features harms the model 
(meaning the drift contains the actual generated signal!), we pivot to Paradigm 2: 
Unsupervised Geometric Features.
Tree models (XGB/CatBoost) only make orthogonal (box) splits. 
By running K-Means clustering and adding the *distances* to all centroids as new features, 
we provide Radial Basis Functions (hyper-spheres). The tree can now split on curves, 
which is highly effective for synthetic data manifolds.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
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
    
    # 1. Base Variables
    df_all['TotalCharges'] = pd.to_numeric(df_all['TotalCharges'], errors='coerce').fillna(0)
    df_all['MonthlyCharges'] = pd.to_numeric(df_all['MonthlyCharges'], errors='coerce').fillna(0)
    df_all['tenure'] = pd.to_numeric(df_all['tenure'], errors='coerce').fillna(0)
    
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
        
    # ==========================================================
    # GEOMETRIC FEATURE ENGINEERING (K-MEANS LATENT DISTANCES)
    # ==========================================================
    print("Generating Unsupervised Geometric Features (KMeans Radial Bases)...")
    numeric_cols = [c for c in df_all.columns if c not in cat_cols]
    
    # Must scale numeric features for KMeans to make sense of euclidean distance
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(df_all[numeric_cols])
    
    # We will use two granularities of K to capture both macro and micro structures
    clusters_configs = [10, 30] 
    
    for k in clusters_configs:
        print(f"  -> Fitting KMeans with K={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        # fit_transform returns the distance to all centroids! This is the "magic"
        distances = kmeans.fit_transform(X_num_scaled) 
        
        # Add these distances as continuous orthogonal features
        for i in range(k):
            df_all[f'kmeans_dist_{k}_c{i}'] = distances[:, i]
            
        # Also add the exact cluster assignment as a categorical variable
        cluster_labels = kmeans.labels_.astype(str)
        cluster_col_name = f'kmeans_cluster_{k}'
        df_all[cluster_col_name] = cluster_labels
        cat_cols.append(cluster_col_name)

    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    print(f"Final Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y, X_test, ids, cat_cols

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, cat_cols = feature_engineering(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  V18 GEOMETRIC MANIFOLDS TRAINING ({NUM_FOLDS}-Fold)")
    print("="*60)
    
    # Tuned V16 hyper-params, plus bumped border_count to handle the highly continuous distance geometries
    best_p = {
        'iterations': 2500,
        'learning_rate': 0.03,
        'depth': 6,
        'l2_leaf_reg': 5.0,
        'random_strength': 1.5,
        'bagging_temperature': 0.5,
        'border_count': 254, # Max GPU precision for euclidean distances
        'eval_metric': 'AUC',
        'random_seed': 42,
        'verbose': False,
        'task_type': 'GPU' # Change to CPU if runtime error occurs
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    test_pool = Pool(X_test, cat_features=cat_cols)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
        valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
        
        model = CatBoostClassifier(**best_p)
        # Using CrossEntropy loss (Logloss is alias). CatBoost natively optimizes well for ranking
        # with these params, while evaluating pure AUC.
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=150, verbose=False)
        
        oof_preds[val_idx] = model.predict_proba(valid_pool)[:, 1]
        test_preds += model.predict_proba(test_pool)[:, 1] / NUM_FOLDS
        print(f"Fold {fold+1} OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}")
        
    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*60)
    print(f"  Final V18 (Geometric Manifolds) OOF AUC: {final_auc:.5f}")
    if final_auc > 0.91757:
        print("  TARGET EXCEEDED! CEILING SHATTERED! > 0.91757")
    print("="*60)
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v18_geometric.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    with open(os.path.join(config.BASE_DIR, 'catboost_v18_report.md'), 'w') as f:
        f.write(f"# V18 Geometric Manifolds\n\n## Final Performance\n- **10-Fold OOF AUC**: **{final_auc:.5f}**\n\n## Pipeline Details\nAdded `{len(X_train.columns) - 15}` Unsupervised KMeans distance features (Radii structural embeddings) spanning multiple K scales.\n")
    
    print("\nDONE! Submission saved to:", sub_prob_path)

if __name__ == "__main__":
    main()
