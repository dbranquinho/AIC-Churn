import os
import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import src.config as config
from src.dataset import load_data, DataProcessor
from src.train_lgb import LGB_MODEL_PATH, LGB_PROCESSOR_PATH

def evaluate_lgb():
    print("Loading LightGBM Processor...")
    processor = DataProcessor.load(LGB_PROCESSOR_PATH)
        
    print("Preparing Test Data based on provided test file...")
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)
    
    print("Loading LightGBM Model...")
    bst = lgb.Booster(model_file=LGB_MODEL_PATH)
    
    print("\nEvaluating...")
    probs = bst.predict(X_test)
    preds = np.where(probs > 0.5, 1, 0)
            
    # Metrics
    print("\n--- LightGBM Evaluation Results ---")
    print(f"Test Set Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Not Churn (0)", "Churn (1)"]))

if __name__ == "__main__":
    evaluate_lgb()
