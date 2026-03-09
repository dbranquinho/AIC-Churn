import os
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
import src.config as config
from src.dataset import load_data, DataProcessor

CAT_MODEL_PATH = os.path.join(config.MODEL_DIR, 'cat_model.cbm')
CAT_PROCESSOR_PATH = os.path.join(config.MODEL_DIR, 'cat_processor.pkl')

def train_cat():
    print("Preparing Training Data for CatBoost...")
    X_train, y_train, processor = load_data(config.TRAIN_DATA_PATH, fit_processor=True)
    processor.save(CAT_PROCESSOR_PATH)
    
    print("\nTraining CatBoost Classifier...")
    clf = CatBoostClassifier(
        iterations=300,
        depth=5,
        learning_rate=0.05,
        loss_function='Logloss',
        random_seed=42,
        verbose=50
    )
    
    clf.fit(X_train, y_train)
    
    train_preds = clf.predict(X_train)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"Training Baseline Accuracy: {train_acc:.4f}")
    
    clf.save_model(CAT_MODEL_PATH)
    print(f"CatBoost model saved to {CAT_MODEL_PATH}")

if __name__ == "__main__":
    train_cat()
