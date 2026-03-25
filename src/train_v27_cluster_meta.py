import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.optimize import minimize
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
        
    cat_cols = [c for c in train_df.columns if c not in numeric_cols]
    
    for col in cat_cols:
        train_df[col] = train_df[col].fillna('Missing').astype(str).astype('category')
        test_df[col] = test_df[col].fillna('Missing').astype(str).astype('category')
        
    return train_df, y, test_df, ids, cat_cols, numeric_cols

def f_roc_auc(weights, oof_xgb, oof_lgb, oof_cat, y):
    blend = weights[0] * oof_xgb + weights[1] * oof_lgb + weights[2] * oof_cat
    return -roc_auc_score(y, blend)

def main():
    print("="*70)
    print(" V27: HYBRID PIPELINE (V25 Tri-Ensemble + V26 Cosine Meta-Features) ")
    print("="*70)
    
    X_train, y, X_test, test_ids, cat_cols, numeric_cols = get_raw_v6_data()
    
    # ---------------------------------------------------------
    # 1. CREATE DISRUPTIVE META-FEATURES (K=3 Cosine Clusters)
    # ---------------------------------------------------------
    print("\nGenerating Cosine Clustering Meta-Features (K=3)...")
    
    # Scikit-learn pipelining needs string types, not pandas category
    train_cluster_df = X_train.copy()
    test_cluster_df = X_test.copy()
    for c in cat_cols:
        train_cluster_df[c] = train_cluster_df[c].astype(str)
        test_cluster_df[c] = test_cluster_df[c].astype(str)
        
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(train_cluster_df)
    X_test_processed = preprocessor.transform(test_cluster_df)
    
    normalizer = Normalizer(norm='l2')
    X_train_norm = normalizer.fit_transform(X_train_processed)
    X_test_norm = normalizer.transform(X_test_processed)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train_norm)
    
    # Extract distances to the 3 personas (centroids)
    train_distances = kmeans.transform(X_train_norm)
    test_distances = kmeans.transform(X_test_norm)
    
    # Extract hard cluster assignments
    train_clusters = kmeans.predict(X_train_norm)
    test_clusters = kmeans.predict(X_test_norm)
    
    # Inject Meta-Features back into the Tree Dataset
    X_train['Cosine_Cluster'] = train_clusters
    X_test['Cosine_Cluster'] = test_clusters
    X_train['Cosine_Cluster'] = X_train['Cosine_Cluster'].astype(str).astype('category')
    X_test['Cosine_Cluster'] = X_test['Cosine_Cluster'].astype(str).astype('category')
    cat_cols_extended = list(cat_cols) + ['Cosine_Cluster']
    
    for i in range(3):
        X_train[f'Dist_to_Cluster_{i}'] = train_distances[:, i]
        X_test[f'Dist_to_Cluster_{i}'] = test_distances[:, i]
        
    print(f"Added Meta-Features: 'Cosine_Cluster' and 3 Distance Columns.")
    print(f"New Training Shape: {X_train.shape}")
    
    # ---------------------------------------------------------
    # 2. TRI-ENSEMBLE NATIVE TRAINING (V25 Logic)
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
    
    print("\nStarting Parallel Native Training (Trees + Meta-Features)...\n")
    
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train.iloc[t_idx], y[t_idx]
        X_va, y_va = X_train.iloc[v_idx], y[v_idx]
        
        # XGBoost
        try:
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2500, early_stopping_rounds=200)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        except Exception:
            xgb_params['device'] = 'cpu'
            m_xgb = xgb.XGBClassifier(**xgb_params, n_estimators=2500, early_stopping_rounds=200)
            m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            
        oof_xgb[v_idx] = m_xgb.predict_proba(X_va)[:, 1]
        test_xgb += m_xgb.predict_proba(X_test)[:, 1] / NUM_FOLDS
        
        # LightGBM
        m_lgb = lgb.LGBMClassifier(**lgb_params, n_estimators=2500)
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric='auc', callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)])
        oof_lgb[v_idx] = m_lgb.predict_proba(X_va)[:, 1]
        test_lgb += m_lgb.predict_proba(X_test)[:, 1] / NUM_FOLDS
        
        # CatBoost
        tr_pool = Pool(X_tr, y_tr, cat_features=cat_cols_extended)
        va_pool = Pool(X_va, y_va, cat_features=cat_cols_extended)
        te_pool = Pool(X_test, cat_features=cat_cols_extended)
        
        m_cat = CatBoostClassifier(**cat_params, iterations=2500)
        m_cat.fit(tr_pool, eval_set=va_pool, early_stopping_rounds=200, verbose=False)
        
        oof_cat[v_idx] = m_cat.predict_proba(va_pool)[:, 1]
        test_cat += m_cat.predict_proba(te_pool)[:, 1] / NUM_FOLDS
        
        print(f"--- FOLD {fold+1} --- XGB: {roc_auc_score(y_va, oof_xgb[v_idx]):.5f} | LGB: {roc_auc_score(y_va, oof_lgb[v_idx]):.5f} | CAT: {roc_auc_score(y_va, oof_cat[v_idx]):.5f}")

    print("\n" + "="*70)
    print("  HILL CLIMBING ENSEMBLE OPTIMIZATION ")
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
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v27_cluster_meta.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"\nFinal V27 (Hybrid) saved to: {sub_path}")

if __name__ == "__main__":
    main()
