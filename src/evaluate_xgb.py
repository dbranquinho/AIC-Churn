import os
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import src.config as config
from src.dataset import load_data, DataProcessor
from src.train_xgb import XGB_MODEL_PATH, XGB_PROCESSOR_PATH

def evaluate_xgb():
    print("Loading XGB Processor...")
    if not os.path.exists(XGB_PROCESSOR_PATH):
        print(f"Error: XGB Processor not found at {XGB_PROCESSOR_PATH}. Run train_xgb.py first.")
        return
    processor = DataProcessor.load(XGB_PROCESSOR_PATH)
        
    print("Preparing Test Data based on provided test file...")
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)
    
    print("Loading XGBoost Model...")
    if not os.path.exists(XGB_MODEL_PATH):
        print(f"Error: XGBoost model not found at {XGB_MODEL_PATH}. Run train_xgb.py first.")
        return
        
    clf = xgb.XGBClassifier()
    clf.load_model(XGB_MODEL_PATH)
    
    print("\nEvaluating...")
    preds = clf.predict(X_test)
            
    # Metrics
    print("\n--- XGBoost Evaluation Results ---")
    print(f"Test Set Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Not Churn (0)", "Churn (1)"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    evaluate_xgb()
