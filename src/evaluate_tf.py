import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import src.config as config
from src.dataset import load_data, DataProcessor
from src.train_tf import TF_MODEL_PATH, TF_PROCESSOR_PATH

def evaluate_tf():
    print("Loading TF Processor...")
    if not os.path.exists(TF_PROCESSOR_PATH):
        print(f"Error: TF Processor not found at {TF_PROCESSOR_PATH}. Run train_tf.py first.")
        return
    processor = DataProcessor.load(TF_PROCESSOR_PATH)
        
    print("Preparing Test Data...")
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)
    
    print("Loading TensorFlow/Keras Model...")
    if not os.path.exists(TF_MODEL_PATH):
        print(f"Error: TF model not found at {TF_MODEL_PATH}. Run train_tf.py first.")
        return
        
    model = keras.models.load_model(TF_MODEL_PATH)
    
    print("\nEvaluating...")
    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)
            
    # Metrics
    print("\n--- TensorFlow/Keras Evaluation Results ---")
    print(f"Test Set Accuracy: {accuracy_score(y_test, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, preds, target_names=["Not Churn (0)", "Churn (1)"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, preds))

if __name__ == "__main__":
    evaluate_tf()
