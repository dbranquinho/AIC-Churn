import os
import torch
import pandas as pd
import src.config as config
from src.dataset import DataProcessor
from src.model import ChurnModel

class ChurnPredictor:
    def __init__(self, model_path=config.MODEL_SAVE_PATH, processor_path=config.PROCESSOR_SAVE_PATH):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load processor
        if not os.path.exists(processor_path):
            raise FileNotFoundError(f"Processor not found at {processor_path}. Did you run train.py?")
        self.processor = DataProcessor.load(processor_path)
        
        # Load model
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Did you run train.py?")
            
        input_dim = self.processor.get_feature_dim()
        self.model = ChurnModel(
            input_dim=input_dim, 
            hidden_units=config.HIDDEN_UNITS, 
            dropout_rate=0.0
        ).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, data):
        """
        Predicts churn for a single record or list of records.
        
        Args:
            data (dict or list of dicts): Raw data containing required features.
            
        Returns:
            list of dicts containing 'prob' (probability of churn) and 'churn' (0 or 1).
        """
        if isinstance(data, dict):
            data = [data]
            
        # 1. Convert to DataFrame
        df = pd.DataFrame(data)
        
        # 2. Transform the raw data
        try:
            X_processed = self.processor.transform(df)
        except Exception as e:
            return {"error": f"Failed during preprocessing. Ensure all expected columns are provided. Details: {e}"}
            
        # 3. Convert to tensor
        X_tensor = torch.tensor(X_processed, dtype=torch.float32).to(self.device)
        
        # 4. Predict
        with torch.no_grad():
            probs = self.model.predict_proba(X_tensor).cpu().numpy().flatten()
            classes = self.model.predict(X_tensor).cpu().numpy().flatten()
            
        # 5. Format results
        results = []
        for p, c in zip(probs, classes):
            results.append({
                "churn_probability": float(p),
                "is_churn": int(c)
            })
            
        return results

if __name__ == "__main__":
    # Example usage:
    print("Initializing predictor...")
    try:
        predictor = ChurnPredictor()
        
        # Sample raw customer data
        sample_customers = [
            {
                "Age": 30,
                "Gender": "Male",
                "Tenure": 39,
                "Usage Frequency": 14,
                "Support Calls": 5,
                "Payment Delay": 18,
                "Subscription Type": "Standard",
                "Contract Length": "Annual",
                "Total Spend": 932,
                "Last Interaction": 17
            },
            {
                "Age": 45,
                "Gender": "Female",
                "Tenure": 2,
                "Usage Frequency": 2,
                "Support Calls": 15,
                "Payment Delay": 30,
                "Subscription Type": "Basic",
                "Contract Length": "Monthly",
                "Total Spend": 100,
                "Last Interaction": 25
            }
        ]
        
        print("\nPredicting on sample customers...")
        predictions = predictor.predict(sample_customers)
        
        for i, pred in enumerate(predictions):
            print(f"Customer {i+1}: Probability of Churn: {pred['churn_probability']:.2%}, Churn Prediction: {'Yes' if pred['is_churn'] else 'No'}")
            
    except Exception as e:
        print(f"Could not run example inference: {e}")
