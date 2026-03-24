import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tempfile
import pandas as pd
from autogluon.tabular import TabularPredictor
import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def main():
    save_path = os.path.join(tempfile.gettempdir(), 'autogluon_v23_churn')
    print(f"Loading fully trained AutoGluon stack from: {save_path}")
    print("This skips retraining and just generates the CSV!")
    
    predictor = TabularPredictor.load(save_path)
    
    print("\nReading test data...")
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    ids = test_df[config.KAGGLE_ID_COL].values
    test_features = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    print("\nGenerating final stacked CSV test predictions (this may take a minute based on the ensemble size)...")
    y_pred_proba = predictor.predict_proba(test_features)
    
    # AutoGluon outputs possibilities for class 0 and 1. We want the positive class (1).
    if 1 in y_pred_proba.columns:
        sub_preds = y_pred_proba[1].values
    elif 'Yes' in y_pred_proba.columns:
        sub_preds = y_pred_proba['Yes'].values
    else:
        # Fallback to the second column if classes were arbitrarily named
        sub_preds = y_pred_proba.iloc[:, 1].values
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: ids,
        config.KAGGLE_TARGET_COL: sub_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v23_autogluon.csv')
    sub_df.to_csv(sub_path, index=False)
    
    print("\n" + "="*70)
    print(f"  DONE! FINAL V23 PREDICTION RECOVERED.")
    print(f"  Upload: {sub_path} to Kaggle!")
    print("="*70)

if __name__ == "__main__":
    main()
