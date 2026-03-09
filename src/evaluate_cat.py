import os
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import src.config as config
from src.dataset import load_data, DataProcessor
from src.train_cat import CAT_MODEL_PATH, CAT_PROCESSOR_PATH

def evaluate_cat():
    print("Loading CatBoost Processor...")
    processor = DataProcessor.load(CAT_PROCESSOR_PATH)
        
    print("Preparing Test Data based on provided test file...")
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)
    
    print("Loading CatBoost Model...")
    clf = CatBoostClassifier()
    clf.load_model(CAT_MODEL_PATH)
    
    print("\nEvaluating...")
    preds = clf.predict(X_test)
            
    # Metrics
    print("\n--- CatBoost Evaluation Results ---")
    print(f"Test Set Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Not Churn (0)", "Churn (1)"]))

if __name__ == "__main__":
    evaluate_cat()
