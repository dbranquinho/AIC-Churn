import torch
import torch.nn as nn

class ChurnModel(nn.Module):
    def __init__(self, input_dim, hidden_units, dropout_rate=0.3):
        """
        A standard Multi-Layer Perceptron (MLP) for tabular data classification.
        
        Args:
            input_dim (int): Number of input features
            hidden_units (list of int): Sizes of hidden layers
            dropout_rate (float): Probability of dropping units to prevent overfitting
        """
        super(ChurnModel, self).__init__()
        
        layers = []
        in_features = input_dim
        
        # Build hidden layers
        for units in hidden_units:
            layers.append(nn.Linear(in_features, units))
            layers.append(nn.BatchNorm1d(units))  # Helps stabilize and speed up training
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            in_features = units
            
        # Output layer
        # Output is a single probability value per sample.
        # We do not use Sigmoid here because we will use BCEWithLogitsLoss during training 
        # which is numerically more stable than Sigmoid + BCELoss.
        layers.append(nn.Linear(in_features, 1))
        
        # Combine all layers into a sequential module
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass.
        """
        return self.network(x)

    def predict_proba(self, x):
        """
        Returns prediction probabilities (0 to 1).
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def predict(self, x, threshold=0.5):
        """
        Returns binary class predictions based on a threshold.
        """
        probs = self.predict_proba(x)
        return (probs >= threshold).float()
