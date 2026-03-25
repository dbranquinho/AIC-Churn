"""
V26 Pipeline - Deep Learning RealMLP (The Final 0.91761 Enabler)
-------------------------------------------------------------------------
Since all Tree-Based methods (AutoGluon, XGB, LGBM, Cat) hit a hard wall at 0.91400,
we must invoke the absolute state-of-the-art Tabular Neural Network architecture used 
by the 1st place competitors: RealMLP.
RealMLP uses SiLU (Swish) activations, strict dropouts (0.05), and strong AdamW 
regularization to bend decision boundaries that trees cannot reach.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)
NUM_FOLDS = 10

# -------------------------------------------------------------
# PYTORCH REAL_MLP ARCHITECTURE
# -------------------------------------------------------------
class RealMLP(nn.Module):
    def __init__(self, input_dim):
        super(RealMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.05),
            
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # We output logits to use BCEWithLogitsLoss for numeric stability
        return self.net(x)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

def get_preprocessed_tensors():
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('float32').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL, config.KAGGLE_ID_COL], errors='ignore')
    
    ids = test_df[config.KAGGLE_ID_COL].values
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    # ---------------------------------------------------------
    # MAGIC SURCHARGE FEATURE (Needed for Max Signal)
    # ---------------------------------------------------------
    for df in [train_df, test_df]:
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
        df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
        df['tenure'] = pd.to_numeric(df['tenure'], errors='coerce').fillna(0)
        df['SurCharge'] = df['TotalCharges'] - (df['tenure'] * df['MonthlyCharges'])
        
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SurCharge']
    cat_cols = [c for c in train_df.columns if c not in numeric_cols]
    
    for col in cat_cols:
        train_df[col] = train_df[col].fillna('Missing').astype(str)
        test_df[col] = test_df[col].fillna('Missing').astype(str)
        
    # Preprocessor for Neural Network
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_encoded = preprocessor.fit_transform(train_df)
    X_test_encoded = preprocessor.transform(test_df)
    
    return X_train_encoded.astype(np.float32), y, X_test_encoded.astype(np.float32), ids

def train_realmlp_fold(X_tr, y_tr, X_va, y_va, input_dim, fold, device):
    # Dataloaders
    train_dataset = TensorDataset(torch.tensor(X_tr), torch.tensor(y_tr).unsqueeze(1))
    val_dataset = TensorDataset(torch.tensor(X_va), torch.tensor(y_va).unsqueeze(1))
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    model = RealMLP(input_dim).to(device)
    
    # Kaggle Magic RealMLP Parameters
    optimizer = optim.AdamW(model.parameters(), lr=0.0075, weight_decay=0.0236)
    criterion = nn.BCEWithLogitsLoss()
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    best_auc = 0.0
    best_weights = None
    patience = 20
    no_improve_epochs = 0
    
    for epoch in range(150):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
        # Validation
        model.eval()
        val_preds = []
        with torch.no_grad():
            for batch_X, _ in val_loader:
                batch_X = batch_X.to(device)
                logits = model(batch_X)
                probs = torch.sigmoid(logits)
                val_preds.extend(probs.cpu().numpy().flatten())
                
        auc = roc_auc_score(y_va, val_preds)
        scheduler.step(auc)
        
        if auc > best_auc:
            best_auc = auc
            best_weights = model.state_dict().copy()
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            
        if no_improve_epochs >= patience:
            break
            
    print(f"  [Fold {fold}] Best PyTorch AUC: {best_auc:.5f}")
    
    model.load_state_dict(best_weights)
    return model

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("="*70)
    print(f" V26 REAL-MLP NEURAL NETWORK (Device: {device}) ")
    print("="*70)
    
    print("Encoding Kaggle Data for Deep Learning...")
    X_train, y, X_test, test_ids = get_preprocessed_tensors()
    input_dim = X_train.shape[1]
    
    skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    
    oof_preds = np.zeros(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    print(f"\nTraining 10-Fold PyTorch MLP (Hidden: 512-256-128, SiLU, AdamW)...")
    for fold, (t_idx, v_idx) in enumerate(skf.split(X_train, y)):
        X_tr, y_tr = X_train[t_idx], y[t_idx]
        X_va, y_va = X_train[v_idx], y[v_idx]
        
        model = train_realmlp_fold(X_tr, y_tr, X_va, y_va, input_dim, fold + 1, device)
        
        # Predict Validation
        model.eval()
        with torch.no_grad():
            va_tensor = torch.tensor(X_va).to(device)
            oof_preds[v_idx] = torch.sigmoid(model(va_tensor)).cpu().numpy().flatten()
            
            # Predict Test
            te_tensor = torch.tensor(X_test).to(device)
            test_preds += torch.sigmoid(model(te_tensor)).cpu().numpy().flatten() / NUM_FOLDS

    final_auc = roc_auc_score(y, oof_preds)
    print("\n" + "="*80)
    print(f"  Final V26 (PyTorch RealMLP) OOF AUC: {final_auc:.5f}")
    print("="*80)
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: test_ids,
        config.KAGGLE_TARGET_COL: test_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v26_realmlp.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"  Neural Network Submission saved to: {sub_path}")
    print("  Use post_process_v22.py to ENSEMBLE this PyTorch output with V24 XGBoost!")

if __name__ == "__main__":
    main()
