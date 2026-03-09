import os
import lightgbm as lgb
from sklearn.metrics import accuracy_score
import src.config as config
from src.dataset import load_data, DataProcessor

LGB_MODEL_PATH = os.path.join(config.MODEL_DIR, 'lgb_model.txt')
LGB_PROCESSOR_PATH = os.path.join(config.MODEL_DIR, 'lgb_processor.pkl')

def train_lgb():
    print("Preparing Training Data for LightGBM...")
    X_train, y_train, processor = load_data(config.TRAIN_DATA_PATH, fit_processor=True)
    processor.save(LGB_PROCESSOR_PATH)
    
    print("\nTraining LightGBM Classifier...")
    clf = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Training Baseline Accuracy: {train_acc:.4f}")
    
    clf.booster_.save_model(LGB_MODEL_PATH)
    print(f"LightGBM model saved to {LGB_MODEL_PATH}")

if __name__ == "__main__":
    train_lgb()
