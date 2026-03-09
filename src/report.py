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
    
    # 3. Load Model
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
    
    # 4. Predict
    all_probs = []
    all_preds = []
    all_labels = []
    
    print("Evaluating Model and generating probabilities...")
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

    # 5. Calculate Metrics
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=["Not Churn (0)", "Churn (1)"], output_dict=True)
    
    # Check if there is only 1 class predicted usually (happens early in training)
    print(f"\nFinal Accuracy: {acc:.4f}")
    
    # Set up plotting directory
    plot_dir = os.path.join(os.path.dirname(config.MODEL_SAVE_PATH), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    
    # --- Plot 1: Confusion Matrix ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Churn', 'Churn'], 
                yticklabels=['Not Churn', 'Churn'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = os.path.join(plot_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Confusion Matrix to {cm_path}")

    # --- Plot 2: ROC Curve ---
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    roc_path = os.path.join(plot_dir, 'roc_curve.png')
    plt.savefig(roc_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved ROC Curve to {roc_path}")

    # --- Plot 3: Precision-Recall Curve ---
    precision, recall, _ = precision_recall_curve(all_labels, all_probs)
    avg_precision = average_precision_score(all_labels, all_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    pr_path = os.path.join(plot_dir, 'pr_curve.png')
    plt.savefig(pr_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved Precision-Recall Curve to {pr_path}")
    
    print("\nMetrics and Plots successfully generated!")

if __name__ == "__main__":
    generate_report()
