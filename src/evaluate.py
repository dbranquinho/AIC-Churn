import torch
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import src.config as config
from src.dataset import load_data, get_dataloader, DataProcessor
from src.model import ChurnModel

def evaluate():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the fitted processor
    try:
        processor = DataProcessor.load(config.PROCESSOR_SAVE_PATH)
    except FileNotFoundError:
        print(f"Error: Processor not found at {config.PROCESSOR_SAVE_PATH}. Run train.py first.")
        return
        
    # 2. Load the test data
    print("Preparing Test Data...")
    X_test, y_test = load_data(config.TEST_DATA_PATH, fit_processor=False, processor=processor)
    test_loader = get_dataloader(X_test, y_test, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. Load Model
    input_dim = processor.get_feature_dim()
    model = ChurnModel(
        input_dim=input_dim, 
        hidden_units=config.HIDDEN_UNITS, 
        dropout_rate=0.0 # Dropout is disabled during evaluation anyway
    ).to(device)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=device))
        print(f"Model weights loaded from {config.MODEL_SAVE_PATH}")
    except FileNotFoundError:
        print(f"Error: Model not found at {config.MODEL_SAVE_PATH}. Run train.py first.")
        return
        
    model.eval()
    
    # 4. Predict
    all_preds = []
    all_labels = []
    
    print("Evaluating...")
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            # Use the threshold-based prediction method
            preds = model.predict(batch_X)
            
            all_preds.extend(preds.cpu().numpy().flatten())
            all_labels.extend(batch_y.numpy().flatten())
            
    # 5. Metrics
    print("\n--- Evaluation Results ---")
    print(f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=["Not Churn (0)", "Churn (1)"]))
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    evaluate()
