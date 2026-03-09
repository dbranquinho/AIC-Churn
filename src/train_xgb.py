import os
import pickle
import xgboost as xgb
from sklearn.metrics import accuracy_score
import src.config as config
from src.dataset import load_data, DataProcessor

# Output path for XGBoost model
XGB_MODEL_PATH = os.path.join(config.MODEL_DIR, 'xgb_model.json')
XGB_PROCESSOR_PATH = os.path.join(config.MODEL_DIR, 'xgb_processor.pkl')

def train_xgb():
    print("Preparing Training Data for XGBoost...")
    # Load and fit identical to neural network
    X_train, y_train, processor = load_data(config.TRAIN_DATA_PATH, fit_processor=True)
    
    processor.save(XGB_PROCESSOR_PATH)
    print(f"XGB Processor saved to {XGB_PROCESSOR_PATH}")
    
    print("\nTraining XGBoost Classifier...")
    # Initialize extreme gradient boosting with deep-learning-esque regularization to prevent overfitting noise
    clf = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5, 
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Check training accuracy (just to ensure it learned *something*)
    train_preds = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Training Baseline Accuracy: {train_acc:.4f}")
    
    # Save the model
    clf.save_model(XGB_MODEL_PATH)
    print(f"XGBoost model saved to {XGB_MODEL_PATH}")

if __name__ == "__main__":
    train_xgb()
