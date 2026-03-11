"""
Kaggle Playground Series S6E3 — Advanced XGBoost v3 Pipeline
Includes extreme feature engineering, KMeans clustering, smoothed OOF targeting, and a deep Optuna search.
"""
import os
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
import optuna
import warnings

warnings.filterwarnings('ignore')

import src.config as config

KAGGLE_MODEL_DIR = os.path.join(config.MODEL_DIR, 'kaggle')
os.makedirs(KAGGLE_MODEL_DIR, exist_ok=True)
SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
REPORT_DIR = os.path.join(config.BASE_DIR, 'models', 'plots')
os.makedirs(REPORT_DIR, exist_ok=True)

NUM_FOLDS = 5

def feature_engineering(train_df, test_df):
    """Apply extreme feature engineering (v3)."""
    print("Applying extreme feature engineering (v3)...")
    
    # Target definition
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0}).values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL])
    
    ids = test_df[config.KAGGLE_ID_COL].values
    train_df = train_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    df_all = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
    train_len = len(train_df)
    
    # --- 1. Feature Crosses ---
    df_all['Contract_Payment'] = df_all['Contract'] + "_" + df_all['PaymentMethod']
    df_all['Internet_Security'] = df_all['InternetService'] + "_" + df_all['OnlineSecurity']
    
    cat_cols = config.KAGGLE_CATEGORICAL_COLS + ['Contract_Payment', 'Internet_Security']
    
    # --- 2. Count/Frequency Encoding ---
    for col in cat_cols:
        freq = df_all[col].value_counts(normalize=True).to_dict()
        df_all[f'{col}_freq'] = df_all[col].map(freq)
        
    # --- 3. Group Statistics ---
    for col in ['Contract', 'PaymentMethod', 'InternetService']:
        grouped = df_all.groupby(col)['MonthlyCharges'].agg(['mean', 'std']).reset_index()
        grouped.columns = [col, f'{col}_Monthly_mean', f'{col}_Monthly_std']
        df_all = df_all.merge(grouped, on=col, how='left')
        
    # --- 4. Total services ---
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'MultipleLines']
    df_all['total_services'] = df_all[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
    
    # --- 5. Financial Features & Ratios (v3) ---
    epsilon = 1e-5
    df_all['avg_charge_per_tenure'] = df_all['TotalCharges'] / (df_all['tenure'] + epsilon)
    df_all['expected_total_charges'] = df_all['tenure'] * df_all['MonthlyCharges']
    df_all['charge_discrepancy'] = df_all['TotalCharges'] - df_all['expected_total_charges']
    
    # New Ratios
    df_all['monthly_over_total'] = df_all['MonthlyCharges'] / (df_all['TotalCharges'] + epsilon)
    df_all['tenure_over_monthly'] = df_all['tenure'] / (df_all['MonthlyCharges'] + epsilon)
    df_all['pct_discrepancy'] = df_all['charge_discrepancy'] / (df_all['expected_total_charges'] + epsilon)
    
    # --- 6. Quantile Binning (v3) ---
    df_all['tenure_bin'] = pd.qcut(df_all['tenure'], q=10, labels=False, duplicates='drop')
    df_all['monthly_bin'] = pd.qcut(df_all['MonthlyCharges'], q=10, labels=False, duplicates='drop')
    df_all['total_bin'] = pd.qcut(df_all['TotalCharges'], q=10, labels=False, duplicates='drop')
    
    # --- 7. KMeans Clustering (v3) ---
    print("Running KMeans clustering for financial profiles...")
    cluster_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
    # Scale briefly for KMeans
    X_clust = (df_all[cluster_features] - df_all[cluster_features].mean()) / df_all[cluster_features].std()
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df_all['financial_cluster'] = kmeans.fit_predict(X_clust).astype(str)
    cat_cols.append('financial_cluster')
    
    # Separate back into train and test
    X_train = df_all.iloc[:train_len].copy()
    X_test = df_all.iloc[train_len:].copy()
    
    # --- 8. Smoothed Out-Of-Fold (OOF) Target Encoding (v3) ---
    oof_cols = ['Contract', 'InternetService', 'PaymentMethod', 'Contract_Payment', 'financial_cluster']
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.KAGGLE_RANDOM_STATE)
    
    smoothing = 10  # Weight given to the global mean (prevents overfitting on small categories)
    
    for col in oof_cols:
        X_train[f'{col}_target_enc'] = 0.0
        X_test[f'{col}_target_enc'] = 0.0
        
        global_mean = y.mean()
        
        for train_idx, val_idx in skf.split(X_train, y):
            X_tr_fold, X_va_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
            y_tr_fold = y[train_idx]
            
            # Calculate counts and means per category
            cat_stats = pd.DataFrame({'y': y_tr_fold, 'cat': X_tr_fold[col].values})
            stats = cat_stats.groupby('cat')['y'].agg(['mean', 'count'])
            
            # Apply Smoothing formula: (count * mean + smoothing * global_mean) / (count + smoothing)
            smoothed_mean = (stats['count'] * stats['mean'] + smoothing * global_mean) / (stats['count'] + smoothing)
            
            # Map to validation fold (add tiny gaussian noise to prevent tree exact splits)
            val_encoded = X_va_fold[col].map(smoothed_mean).fillna(global_mean)
            noise = np.random.normal(0, 0.005, size=len(val_encoded))
            X_train.loc[X_train.index[val_idx], f'{col}_target_enc'] = val_encoded + noise
            
        # For the test set, we map using the smoothed means calculated on the ENTIRE training set
        cat_stats_full = pd.DataFrame({'y': y, 'cat': X_train[col].values})
        stats_full = cat_stats_full.groupby('cat')['y'].agg(['mean', 'count'])
        smoothed_mean_full = (stats_full['count'] * stats_full['mean'] + smoothing * global_mean) / (stats_full['count'] + smoothing)
        
        X_test[f'{col}_target_enc'] = X_test[col].map(smoothed_mean_full).fillna(global_mean)
        
    # --- 9. Final Categorical Casting ---
    for col in cat_cols:
        X_train[col] = X_train[col].astype('category')
        X_test[col] = X_test[col].astype('category')
        
    print(f"Final Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    return X_train, y, X_test, ids

def objective(trial, X, y):
    """Optuna objective using an aggressive deep search space (v3)."""
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',  # Enforce GPU usage
        'enable_categorical': True,
        'random_state': config.KAGGLE_RANDOM_STATE,
        # We enforce a lower learning rate for better generalization
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 25),
        'subsample': trial.suggest_float('subsample', 0.4, 0.95),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 0.95),
        'gamma': trial.suggest_float('gamma', 0, 10),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 50, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 50, log=True),
        'max_bin': trial.suggest_int('max_bin', 256, 1024),
        'max_cat_to_onehot': trial.suggest_int('max_cat_to_onehot', 1, 10),
    }

    # Fast 80/20 split
    X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=config.KAGGLE_RANDOM_STATE)
    
    dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
    
    # We increase early stopping slightly due to lower learning rates
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=2500,
        evals=[(dvalid, 'validation')],
        early_stopping_rounds=100,
        verbose_eval=False
    )
    
    return model.best_score

def main():
    print("Loading data...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    X_train, y, X_test, test_ids = feature_engineering(train_df, test_df)
    
    print("\n" + "="*60)
    print("  Phase 1: Deep Optuna Hyperparameter Tuning (80 Trials)")
    print("="*60)
    
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=15)
    study = optuna.create_study(direction="maximize", pruner=pruner)
    
    n_trials = 80
    print(f"Running {n_trials} deep trials (this will take longer, but yields better parameters)...")
    study.optimize(lambda trial: objective(trial, X_train, y), n_trials=n_trials, n_jobs=1)
    
    print("\nUltimate Best Tuning Params:")
    best_params = study.best_trial.params
    for k, v in best_params.items():
        print(f"  {k}: {v}")
        
    best_params.update({
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'hist',
        'device': 'cuda',  # Enforce GPU usage
        'enable_categorical': True,
        'random_state': config.KAGGLE_RANDOM_STATE
    })
    
    print("\n" + "="*60)
    print("  Phase 2: 5-Fold Ensembling")
    print("="*60)
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=config.KAGGLE_RANDOM_STATE)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    feature_importance_acc = {}
    
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y)):
        print(f"\n--- Training Fold {fold+1}/{NUM_FOLDS} ---")
        X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
        X_va, y_va = X_train.iloc[val_idx], y[val_idx]
        
        dtrain = xgb.DMatrix(X_tr, label=y_tr, enable_categorical=True)
        dvalid = xgb.DMatrix(X_va, label=y_va, enable_categorical=True)
        
        # Train fold model
        model = xgb.train(
            best_params,
            dtrain,
            num_boost_round=3500,
            evals=[(dtrain, 'train'), (dvalid, 'valid')],
            early_stopping_rounds=150,
            verbose_eval=100
        )
        
        # Save validation predictions for OOF score
        oof_preds[val_idx] = model.predict(dvalid)
        
        # Predict on Test Set (Accumulate probabilities for average ensemble)
        test_preds += model.predict(dtest) / NUM_FOLDS
        
        # Accumulate feature importance
        imp = model.get_score(importance_type='weight')
        for k, v in imp.items():
            feature_importance_acc[k] = feature_importance_acc.get(k, 0) + v
            
        # Save fold model
        model_path = os.path.join(KAGGLE_MODEL_DIR, f'xgb_v3_fold_{fold}.json')
        model.save_model(model_path)
    
    # Calculate Overall OOF Score
    oof_auc = roc_auc_score(y, oof_preds)
    print("\n" + "="*60)
    print(f"  Final Out-Of-Fold (OOF) AUC Score: {oof_auc:.5f}")
    print("="*60)
    
    print("\nGenerating Submissions...")
    # Create probability submission
    sub_prob = pd.DataFrame({'id': test_ids, 'Churn': test_preds})
    sub_prob_path = os.path.join(SUBMISSION_DIR, 'submission_xgb_v3_ensemble.csv')
    sub_prob.to_csv(sub_prob_path, index=False)
    
    print(f"Submission saved -> {sub_prob_path}")
    
    # Generate Report
    print("\nGenerating Report...")
    import matplotlib.pyplot as plt
    
    feat_imp_df = pd.DataFrame({
        'Feature': list(feature_importance_acc.keys()),
        'Importance (Sum over 5 Folds)': list(feature_importance_acc.values())
    }).sort_values('Importance (Sum over 5 Folds)', ascending=False)
    
    # Plot top 20
    plt.figure(figsize=(12, 8))
    top_20 = feat_imp_df.head(20)
    plt.barh(top_20['Feature'][::-1], top_20['Importance (Sum over 5 Folds)'][::-1], color='violet')
    plt.xlabel('XGBoost Relative Feature Weight')
    plt.title('Top 20 Features - XGBoost v3 (Extreme Profile)')
    plt.tight_layout()
    plot_path = os.path.join(REPORT_DIR, 'xgb_v3_feature_importance.png')
    plt.savefig(plot_path)
    
    report_content = f"""# XGBoost v3 (Target-Breaker) Report

## Strategy & Extreme Features
This script was rewritten aggressively to break the 0.91752 performance barrier:
1. **Financial Ratios & Profiles**: Added KMeans clustering ($k=8$), percentile binning, and cross-ratios (`Monthly / Total`).
2. **Smoothed Target Encoding**: OOF encoding is now smoothed and Gaussian-noised to prevent high-cardinality leakage.
3. **Deep Optuna Optimization (80 Trials)**: Bound learning rate to `[0.005, 0.05]` to force slow, highly accurate tree growth.
4. **Ensembling**: A robust 5-Fold Stratified Average.

## Final Performance
- **Out-Of-Fold (OOF) ROC-AUC**: **{oof_auc:.5f}**

## Top 15 Most Important Features
| Feature | Importance Score (Ensemble Sum) |
|---------|--------------------------------|
"""
    for _, row in feat_imp_df.head(15).iterrows():
        report_content += f"| {row['Feature']} | {row['Importance (Sum over 5 Folds)']:.1f} |\n"
        
    report_path = os.path.join(config.BASE_DIR, 'xgb_v3_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print(f"Report saved -> {report_path}")
    print("\nDONE! You can upload submission_xgb_v3_ensemble.csv to Kaggle.")

if __name__ == "__main__":
    main()
