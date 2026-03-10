import os

# Base directory mapping
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# Ensure models directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Datasets
TRAIN_DATA_PATH = os.path.join(DATA_DIR, 'customer_churn_dataset-training-master.csv')
TEST_DATA_PATH = os.path.join(DATA_DIR, 'customer_churn_dataset-testing-master.csv')

# Outputs
MODEL_SAVE_PATH = os.path.join(MODEL_DIR, 'churn_model.pth')
PROCESSOR_SAVE_PATH = os.path.join(MODEL_DIR, 'processor.pkl')

# Feature definitions
TARGET_COL = 'Churn'
ID_COL = 'CustomerID'

NUMERICAL_COLS = [
    'Age', 
    'Tenure', 
    'Usage Frequency', 
    'Support Calls', 
    'Payment Delay', 
    'Total Spend', 
    'Last Interaction'
]

CATEGORICAL_COLS = [
    'Gender', 
    'Subscription Type', 
    'Contract Length'
]

# Neural Network Hyperparameters
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_UNITS = [128, 64, 32]
DROPOUT_RATE = 0.3

# ============================================================
# Kaggle Playground Series S6E3 - Telecom Churn
# ============================================================

# Kaggle Datasets
KAGGLE_TRAIN_PATH = os.path.join(DATA_DIR, 'train.csv')
KAGGLE_TEST_PATH = os.path.join(DATA_DIR, 'test.csv')
KAGGLE_SAMPLE_SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')

# Kaggle Feature Definitions
KAGGLE_ID_COL = 'id'
KAGGLE_TARGET_COL = 'Churn'

KAGGLE_NUMERICAL_COLS = [
    'tenure',
    'SeniorCitizen',
    'MonthlyCharges',
    'TotalCharges',
]

KAGGLE_CATEGORICAL_COLS = [
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',
]

# Kaggle Neural Network Hyperparameters
KAGGLE_BATCH_SIZE = 256
KAGGLE_LEARNING_RATE = 0.001
KAGGLE_EPOCHS = 15
KAGGLE_HIDDEN_UNITS = [256, 128, 64]
KAGGLE_DROPOUT_RATE = 0.3

# Cross-validation
KAGGLE_N_FOLDS = 5
KAGGLE_RANDOM_STATE = 42
