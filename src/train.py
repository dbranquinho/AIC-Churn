import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import src.config as config
from src.dataset import load_data, get_dataloader
from src.model import ChurnModel

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load and process training data
    print("Preparing Training Data...")
    X_train, y_train, processor = load_data(config.TRAIN_DATA_PATH, fit_processor=True)
    
    # Save the fitted processor so we can use it in evaluation/inference
    processor.save(config.PROCESSOR_SAVE_PATH)
    print(f"Processor saved to {config.PROCESSOR_SAVE_PATH}")
    
    train_loader = get_dataloader(X_train, y_train, config.BATCH_SIZE, shuffle=True)
    
    # 2. Init Model
    input_dim = processor.get_feature_dim()
    model = ChurnModel(
        input_dim=input_dim, 
        hidden_units=config.HIDDEN_UNITS, 
        dropout_rate=config.DROPOUT_RATE
    ).to(device)
    
    # 3. Optimization and Loss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    
    # 4. Training Loop
    print("\nStarting Training...")
    best_loss = float('inf')
    
    for epoch in range(config.EPOCHS):
        model.train()
        epoch_loss = 0.0
        
        # Use tqdm for a progress bar
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}")
        
        for batch_X, batch_y in progress_bar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.EPOCHS}], Average Loss: {avg_loss:.4f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"Model saved to {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    train()
