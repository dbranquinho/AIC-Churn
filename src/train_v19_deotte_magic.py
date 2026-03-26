"""
V19 Pipeline - Chris Deotte's "PAIR TE" Magic 
This implements the top Kaggle Grandmaster approach for Playground S6E3:
1. Bin continuous features into categories.
2. Combine all 19 raw features into exactly 171 pairwise categorical combinations.
3. Target Encode all 171 pairs strictly inside the Cross-Validation folds to eliminate leakage.
4. Train GPU CatBoost on this high-dimensional target-encoded matrix.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from category_encoders import TargetEncoder
import itertools
from catboost import CatBoostClassifier, Pool
import warnings

warnings.filterwarnings('ignore')

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
    
    # 2. Adaptive Binning of continuous variables (Chris Deotte recommends binning for PAIR TE)
    print("Binning numeric features...")
    df_all['tenure_binned'] = pd.qcut(df_all['tenure'], q=15, duplicates='drop').astype(str)
    df_all['MonthlyCharges_binned'] = pd.qcut(df_all['MonthlyCharges'], q=20, duplicates='drop').astype(str)
    df_all['TotalCharges_binned'] = pd.qcut(df_all['TotalCharges'], q=20, duplicates='drop').astype(str)
    
    # Categoricals for pairs (excluding the raw continuos ones)
    cat_features = [f for f in base_features if f not in ['tenure', 'MonthlyCharges', 'TotalCharges']]
    
    all_binned_and_cats = cat_features + ['tenure_binned', 'MonthlyCharges_binned', 'TotalCharges_binned']
    
    # Fill missing before string concat
    for col in all_binned_and_cats:
        df_all[col] = df_all[col].fillna("Missing").astype(str)
        
    # 3. Generate exactly 171 Pairwise String combinations
    print(f"Generating 171 PAIR combinations...")
    pair_cols = []
    
    for col1, col2 in itertools.combinations(all_binned_and_cats, 2):
        pair_name = f"PAIR_{col1}_x_{col2}"
        df_all[pair_name] = df_all[col1] + "_X_" + df_all[col2]
        pair_cols.append(pair_name)
        
    print(f"Total shape after 171 pairs generation: {df_all.shape}")
        
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    # We will pass the RAW categoricals natively to CatBoost, but the 171 Pairs will be Target Encoded
    raw_cat_cols = cat_features + ['tenure_binned', 'MonthlyCharges_binned', 'TotalCharges_binned']
    
    return X_train, y, X_test, ids, pair_cols, raw_cat_cols

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids, pair_cols, raw_cat_cols = prepare_chris_deotte_features(train_df, test_df)
    
    print("\n" + "="*60)
    print(f"  V19 CHRIS DEOTTE'S PAIR TE MAGIC ({NUM_FOLDS}-Fold)")
    print("="*60)
    
    # Robust baseline CatBoost Params
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
        'task_type': 'GPU'
    }
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx].copy(), y[train_idx]
        X_va, y_va = X_train.iloc[val_idx].copy(), y[val_idx]
        X_te = X_test.copy()
        
        # MAGIC: In-Fold Target Encoding for all 171 Pairs
        # Smoothing=20.0 prevents overfitting on rare categorical intersections
        print(f"  -> Target Encoding 171 Pairs (Smoothing 20.0)...")
        te = TargetEncoder(cols=pair_cols, smoothing=20.0)
        
        # Fit on training data ONLY
        X_tr[pair_cols] = te.fit_transform(X_tr[pair_cols], y_tr)
        
        # Transform Validation and Test
        X_va[pair_cols] = te.transform(X_va[pair_cols])
        X_te[pair_cols] = te.transform(X_te[pair_cols])
        
        # Now X_tr[pair_cols] contains 171 continuous probabilities!
        
        train_pool = Pool(X_tr, y_tr, cat_features=raw_cat_cols)
        valid_pool = Pool(X_va, y_va, cat_features=raw_cat_cols)
        test_pool = Pool(X_te, cat_features=raw_cat_cols)
        
        print("  -> Training CatBoost...")
        model = CatBoostClassifier(**best_p)
        model.fit(train_pool, eval_set=valid_pool, early_stopping_rounds=150, verbose=False)
        
        oof_preds[val_idx] = model.predict_proba(valid_pool)[:, 1]
        test_preds += model.predict_proba(test_pool)[:, 1] / NUM_FOLDS
        
        print(f"  -> Fold {fold+1} OOF AUC: {roc_auc_score(y_va, oof_preds[val_idx]):.5f}\n")
        
    final_auc = roc_auc_score(y, oof_preds)
    
    print("\n" + "="*80)
    print(f"  Final V19 (Chris Deotte PAIR TE) OOF AUC: {final_auc:.5f}")
    if final_auc > 0.91757:
        print("  !!! CHRIS DEOTTE MAGIC WORKED! CEILING BROKEN !!!")
    print("="*80)
    
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_v19_deotte_magic.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    with open(os.path.join(config.BASE_DIR, 'catboost_v19_report.md'), 'w') as f:
        f.write(f"# V19 Chris Deotte PAIR TE Magic\n\n## Final Performance\n- **10-Fold OOF AUC**: **{final_auc:.5f}**\n\n## Pipeline Details\nGenerated exactly 171 Binned Pairwise Features and strict in-fold Target Encoding to bypass synthetic manifold ceilings.\n")
    
    print("\nDONE! Submission saved to:", sub_prob_path)

if __name__ == "__main__":
    main()
