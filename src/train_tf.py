import os
import numpy as np
from sklearn.metrics import accuracy_score
import src.config as config
from src.dataset import load_data, DataProcessor
from src.model_tf import build_keras_model

# Output paths for TensorFlow model
TF_MODEL_PATH = os.path.join(config.MODEL_DIR, 'tf_churn_model.keras')
TF_PROCESSOR_PATH = os.path.join(config.MODEL_DIR, 'tf_processor.pkl')

def train_tf():
    print("Preparing Training Data for TensorFlow/Keras...")
    X_train, y_train, processor = load_data(config.TRAIN_DATA_PATH, fit_processor=True)
    
    processor.save(TF_PROCESSOR_PATH)
    print(f"TF Processor saved to {TF_PROCESSOR_PATH}")
    
    # Build model
    input_dim = processor.get_feature_dim()
    model = build_keras_model(
        input_dim=input_dim,
        hidden_units=config.HIDDEN_UNITS,
        dropout_rate=config.DROPOUT_RATE
    )
    
    model.summary()
    
    print("\nTraining TensorFlow/Keras MLP...")
    
    # Use EarlyStopping to avoid overfitting
    from tensorflow.keras.callbacks import EarlyStopping
    early_stop = EarlyStopping(
        monitor='loss',
        patience=3,
        restore_best_weights=True,
        verbose=1
    )
    
    model.fit(
        X_train, y_train,
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Training accuracy
    train_probs = model.predict(X_train, verbose=0).flatten()
    train_preds = (train_probs >= 0.5).astype(int)
    train_acc = accuracy_score(y_train, train_preds)
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    
    # Save model
    model.save(TF_MODEL_PATH)
    print(f"TensorFlow model saved to {TF_MODEL_PATH}")

if __name__ == "__main__":
    train_tf()
