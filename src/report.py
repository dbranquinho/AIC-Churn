import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
import src.config as config
from src.dataset import load_data, get_dataloader, DataProcessor
from src.model import ChurnModel


def _load_tf_predictions(X_test):
    """Attempt to load the TensorFlow model and return its probabilities."""
    try:
        from tensorflow import keras
        from src.train_tf import TF_MODEL_PATH, TF_PROCESSOR_PATH

        if not os.path.exists(TF_MODEL_PATH):
            print("TensorFlow model not found — skipping TF curves.")
            return None

        tf_model = keras.models.load_model(TF_MODEL_PATH)

        # TF model uses its own processor, but shares the same preprocessing logic.
        # We re-use X_test already preprocessed by the PyTorch processor since the
        # DataProcessor is identical. If a separate TF processor was saved we could
        # load it, but the feature pipeline is deterministic and identical.
        tf_probs = tf_model.predict(X_test, verbose=0).flatten()
        return tf_probs
    except ImportError:
        print("TensorFlow not installed — skipping TF curves.")
        return None
    except Exception as e:
        print(f"Could not load TensorFlow model: {e} — skipping TF curves.")
        return None


def generate_report():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Load the fitted processor
    if not os.path.exists(config.PROCESSOR_SAVE_PATH):
        print(f"Error: Processor not found at {config.PROCESSOR_SAVE_PATH}.")
        return
    processor = DataProcessor.load(config.PROCESSOR_SAVE_PATH)

    # 2. Load the test data
    print("Preparing Test Data for Report...")
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)
    test_loader = get_dataloader(X_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)

    # 3. Load PyTorch Model
    input_dim = processor.get_feature_dim()
    model = ChurnModel(
        input_dim=input_dim,
        hidden_units=config.HIDDEN_UNITS,
        dropout_rate=0.0
    ).to(device)

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Error: Model not found at {config.MODEL_SAVE_PATH}.")
        return

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
    model.eval()

    # 4. PyTorch predictions
    all_probs = []
    all_preds = []
    all_labels = []

    print("Evaluating PyTorch Model...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            probs = model.predict_proba(batch_X)
            preds = (probs >= 0.5).float()

            all_probs.extend(probs.cpu().numpy().flatten())
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.numpy().flatten())

    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # 5. TensorFlow predictions (optional — only if model exists)
    print("Attempting to load TensorFlow/Keras model...")
    tf_probs = _load_tf_predictions(X_test)

    # 6. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"\nPyTorch MLP Accuracy: {acc:.4f}")

    if tf_probs is not None:
        tf_preds = (tf_probs >= 0.5).astype(int)
        tf_acc = accuracy_score(all_labels, tf_preds)
        print(f"TensorFlow/Keras MLP Accuracy: {tf_acc:.4f}")

    # Set up assets directory (project root/assets)
    assets_dir = os.path.join(config.BASE_DIR, 'assets')
    os.makedirs(assets_dir, exist_ok=True)

    # --- Plot 1: Confusion Matrix (PyTorch) ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Churn', 'Churn'],
                yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix — PyTorch MLP')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(assets_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

    # --- Plot 2: ROC Curve (Comparative) ---
    fpr_pt, tpr_pt, _ = roc_curve(all_labels, all_probs)
    roc_auc_pt = auc(fpr_pt, tpr_pt)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr_pt, tpr_pt, color='darkorange', lw=2,
             label=f'PyTorch MLP (AUC = {roc_auc_pt:.3f})')

    if tf_probs is not None:
        fpr_tf, tpr_tf, _ = roc_curve(all_labels, tf_probs)
        roc_auc_tf = auc(fpr_tf, tpr_tf)
        plt.plot(fpr_tf, tpr_tf, color='green', lw=2,
                 label=f'TensorFlow/Keras MLP (AUC = {roc_auc_tf:.3f})')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) — Model Comparison')
    plt.legend(loc="lower right")
    roc_path = os.path.join(assets_dir, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")

    # --- Plot 3: Precision-Recall Curve (Comparative) ---
    precision_pt, recall_pt, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision_pt = average_precision_score(all_labels, all_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_pt, precision_pt, color='purple', lw=2,
             label=f'PyTorch MLP (AP = {avg_precision_pt:.3f})')

    if tf_probs is not None:
        precision_tf, recall_tf, _ = precision_recall_curve(all_labels, tf_probs)
        avg_precision_tf = average_precision_score(all_labels, tf_probs)
        plt.plot(recall_tf, precision_tf, color='teal', lw=2,
                 label=f'TensorFlow/Keras MLP (AP = {avg_precision_tf:.3f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve — Model Comparison')
    plt.legend(loc="lower left")
    pr_path = os.path.join(assets_dir, 'pr_curve.png')
    plt.savefig(pr_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Precision-Recall Curve to {pr_path}")

    print("\nMetrics and Plots successfully generated!")


if __name__ == "__main__":
    generate_report()
