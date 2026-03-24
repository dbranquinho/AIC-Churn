"""
V23 Pipeline - The AutoGluon "Kaggle Grandmaster Cheat"
When all manual implementations (Target Encoding, Ensembles, Hill Climbing)
hit an unbreakable mathematical wall (0.913-0.915 LB), Kaggle winners use AutoGluon.
This single script triggers AWS's AutoML which trains Neural Nets, Random Forests,
CatBoost, XGBoost, and LightGBM across heavily regularized native stacked 
ensembles automatically.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    from autogluon.tabular import TabularPredictor
except ImportError:
    print("\n[CRITICAL ERROR] AutoGluon is not installed.")
    print("Por favor, rode o comando abaixo no terminal da sua VENV antes de continuar:")
    print("pip install autogluon")
    sys.exit(1)

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def main():
    print("="*70)
    print(" V23 THE AUTOGLUON MATADOR (Kaggle 'Best Quality' Preset) ")
    print("="*70)
    
    print("\nLoading raw unadulterated dataset...")
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    # Fix target specifically to binary integers natively to ensure prediction is prob(1)
    train_df[config.KAGGLE_TARGET_COL] = train_df[config.KAGGLE_TARGET_COL].map(
        {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}
    ).fillna(0).astype('int64')
    
    # Drop IDs 
    if config.KAGGLE_ID_COL in train_df.columns:
        train_df = train_df.drop(columns=[config.KAGGLE_ID_COL])
        
    ids = test_df[config.KAGGLE_ID_COL].values
    test_features = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    # -------------------------------------------------------------
    # AUTO GLUON MAGIC
    # -------------------------------------------------------------
    print("\n[AutoGluon] Initialization. Setting target metric to ROC_AUC...")
    import tempfile
    save_path = os.path.join(tempfile.gettempdir(), 'autogluon_v23_churn')
    print(f"[AutoGluon] Models will be temporarily saved to: {save_path} to avoid OneDrive sync locks.")
    
    # We ask AutoGluon to maximize roc_auc directly
    predictor = TabularPredictor(
        label=config.KAGGLE_TARGET_COL,
        eval_metric='roc_auc',
        path=save_path,
        verbosity=2  # Level 2 shows training summary
    )
    
    print("\n[AutoGluon] Training the preset='best_quality'. This will take time.")
    print("            It will train FastAI, Tabular PyTorch, LightGBM, XGBoost, ")
    print("            CatBoost, KNN, and ensemble stack them automatically.")
    
    predictor.fit(
        train_data=train_df,
        presets='best_quality',  # Kaggle Grandmaster preset
        time_limit=3600 * 2,     # Limit to 2 hours runtime max (it will stop safely if hit)
        num_gpus=1               # Utilize your GPUs automatically!
    )
    
    # -------------------------------------------------------------
    # PREDICTION & LEADERBOARD
    # -------------------------------------------------------------
    print("\n" + "="*70)
    print("  AUTOGLUON OOF LEADERBOARD SUMMARY ")
    print("="*70)
    lb = predictor.leaderboard(train_df, silent=True)
    print(lb.head(15).to_string())
    
    # The ultimate ensemble model (stacker) is selected automatically
    best_autogluon_model = predictor.model_best
    print(f"\n[AutoGluon] Ultimate Meta-Model Selected: {best_autogluon_model}")
    
    print("\nGenerating final stacked CSV test predictions...")
    y_pred_proba = predictor.predict_proba(test_features)
    
    # Probability of class 1 (Churn -> Yes)
    # y_pred_proba is a Pandas DataFrame with columns [0, 1]
    sub_preds = y_pred_proba[1].values
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: ids,
        config.KAGGLE_TARGET_COL: sub_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v23_autogluon.csv')
    sub_df.to_csv(sub_path, index=False)
    
    print("\n" + "="*70)
    print(f"  DONE! FINAL V23 SCRIPT FINISHED.")
    print(f"  Upload: {sub_path} to Kaggle!")
    print("="*70)

if __name__ == "__main__":
    main()
