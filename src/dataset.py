import pandas as pd
import numpy as np
import torch
import pickle
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer

# Local Imports
import src.config as config

class ChurnDataset(Dataset):
    def __init__(self, X, y):
        """
        Args:
            X (np.ndarray): Processed feature matrix
            y (np.ndarray): Target labels
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            self.y = None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

class DataProcessor:
    def __init__(self):
        """
        Initializes the scikit-learn preprocessing pipeline.
        """
        # We use ColumnTransformer to apply different scaling to different columns
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), config.NUMERICAL_COLS),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), config.CATEGORICAL_COLS)
            ]
        )
        self.is_fitted = False

    def fit_transform(self, df):
        """
        Fits the preprocessor on the training dataframe and transforms it.
        Handles missing values before transforming.
        """
        df = self._clean_data(df)
        
        # Fit and transform
        X_processed = self.preprocessor.fit_transform(df)
        self.is_fitted = True
        return X_processed

    def transform(self, df):
        """
        Transforms a dataframe using the fitted preprocessor.
        """
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before calling transform().")
            
        df = self._clean_data(df)
        X_processed = self.preprocessor.transform(df)
        return X_processed

    def _clean_data(self, df):
        """
        Basic cleaning logic: remove ID column, drop NaN rows since simple imputation 
        might not fit all columns uniformly for a churn dataset without deeper analysis.
        """
        # Make a copy to avoid SettingWithCopyWarning
        df_clean = df.copy()
        
        if config.ID_COL in df_clean.columns:
            df_clean = df_clean.drop(columns=[config.ID_COL])
            
        # Drop rows with NaN in features we care about
        cols_to_check = config.NUMERICAL_COLS + config.CATEGORICAL_COLS
        cols_present = [col for col in cols_to_check if col in df_clean.columns]
        df_clean = df_clean.dropna(subset=cols_present)
        
        return df_clean

    def get_feature_dim(self):
        """
        Returns the output dimension of the engineered features.
        Useful for building the input layer of the PyTorch neural network.
        """
        if not self.is_fitted:
            raise ValueError("Must fit before retrieving dimension.")
        
        # Calculate feature dimension dynamically based on OneHotEncoder outputs
        # Scaler outputs the same number as numerical cols
        num_dim = len(config.NUMERICAL_COLS)
        
        # OrdinalEncoder outputs exactly same number of cols as input config.CATEGORICAL_COLS
        cat_dim = len(config.CATEGORICAL_COLS)
        
        return num_dim + cat_dim
    
    def save(self, path):
        """Saves the fitted processor."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)
            
    @classmethod
    def load(cls, path):
        """Loads a fitted processor."""
        with open(path, 'rb') as f:
            return pickle.load(f)

def load_data(filepath, fit_processor=False, processor=None):
    """
    Loads data from CSV and applies preprocessing.
    """
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # We drop any NaNs in the target variable
    if config.TARGET_COL in df.columns:
        df = df.dropna(subset=[config.TARGET_COL])
        y = df[config.TARGET_COL].values
    else:
        y = None
        
    if fit_processor:
        processor = DataProcessor()
        X = processor.fit_transform(df)
        return X, y, processor
    else:
        # We might have dropped rows in X during transform due to NaNs in features, 
        # so we need to ensure y matches those rows.
        # The easiest way is to let the processor clean the dataframe first.
        df_clean = processor._clean_data(df)
        
        if config.TARGET_COL in df_clean.columns:
            y = df_clean[config.TARGET_COL].values
        
        X = processor.transform(df_clean)
        return X, y

def get_dataloader(X, y, batch_size, shuffle):
    """
    Returns a PyTorch DataLoader.
    """
    dataset = ChurnDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# ============================================================
# Kaggle Playground Series S6E3 - Telecom Churn
# ============================================================

class KaggleDataProcessor:
    """Preprocessor for the Kaggle telecom churn dataset."""
    
    def __init__(self):
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), config.KAGGLE_NUMERICAL_COLS),
                ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), config.KAGGLE_CATEGORICAL_COLS)
            ]
        )
        self.is_fitted = False

    def fit_transform(self, df):
        df = self._clean_data(df)
        X_processed = self.preprocessor.fit_transform(df)
        self.is_fitted = True
        return X_processed

    def transform(self, df):
        if not self.is_fitted:
            raise ValueError("Processor must be fitted before calling transform().")
        df = self._clean_data(df)
        return self.preprocessor.transform(df)

    def _clean_data(self, df):
        df_clean = df.copy()
        # Drop id column
        if config.KAGGLE_ID_COL in df_clean.columns:
            df_clean = df_clean.drop(columns=[config.KAGGLE_ID_COL])
        # Drop target column if present (we extract it separately)
        if config.KAGGLE_TARGET_COL in df_clean.columns:
            df_clean = df_clean.drop(columns=[config.KAGGLE_TARGET_COL])
        return df_clean

    def get_feature_dim(self):
        if not self.is_fitted:
            raise ValueError("Must fit before retrieving dimension.")
        num_dim = len(config.KAGGLE_NUMERICAL_COLS)
        # OrdinalEncoder outputs exactly same number of cols as input config.KAGGLE_CATEGORICAL_COLS
        cat_dim = len(config.KAGGLE_CATEGORICAL_COLS)
        return num_dim + cat_dim

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as f:
            return pickle.load(f)


def kaggle_load_data(filepath, fit_processor=False, processor=None):
    """
    Loads Kaggle telecom churn data. 
    Encodes target: 'Yes' -> 1, 'No' -> 0.
    Returns (X, y, processor) if fit_processor=True, else (X, y, ids).
    """
    print(f"Loading Kaggle data from {filepath}...")
    df = pd.read_csv(filepath)
    
    # Preserve ids for submission
    ids = df[config.KAGGLE_ID_COL].values if config.KAGGLE_ID_COL in df.columns else None
    
    # Encode target
    if config.KAGGLE_TARGET_COL in df.columns:
        y = df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0}).values
    else:
        y = None

    if fit_processor:
        processor = KaggleDataProcessor()
        X = processor.fit_transform(df)
        return X, y, processor
    else:
        X = processor.transform(df)
        return X, y, ids
