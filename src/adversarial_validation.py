import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import src.config as config
from src.dataset import DataProcessor

def run_adversarial_validation():
    print("Loading Training and Testing Data for Adversarial Validation...")
    df_train = pd.read_csv(config.TRAIN_DATA_PATH)
    df_test = pd.read_csv(config.TEST_DATA_PATH)
    
    # Drop the target column (Churn) because we want to see if the *features* are different
    df_train = df_train.drop(columns=[config.TARGET_COL])
    df_test = df_test.drop(columns=[config.TARGET_COL])
    
    # Drop CustomerID as it's just an index
    if 'CustomerID' in df_train.columns:
        df_train = df_train.drop(columns=['CustomerID'])
    if 'CustomerID' in df_test.columns:
        df_test = df_test.drop(columns=['CustomerID'])
        
    print(f"Training shape: {df_train.shape}, Testing shape: {df_test.shape}")
    
    # Create the adversarial target: 0 for train, 1 for test
    df_train['is_test'] = 0
    df_test['is_test'] = 1
    
    # Combine datasets
    df_combined = pd.concat([df_train, df_test], ignore_index=True)
    df_combined = df_combined.dropna()
    
    # Process the combined data (scaling, encoding)
    processor = DataProcessor()
    
    # We fit_transform on everything to prepare the features for the adversarial test
    X = processor.fit_transform(df_combined.drop(columns=['is_test']))
    y = df_combined['is_test'].values
    
    print("\nSplitting mixed data to train an Adversarial XGBoost...")
    # Split the *combined* dataset to see if we can identify which row is which
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Train XGBoost to distinguish between Train and Test rows
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    
    print("Training...")
    clf.fit(X_train, y_train)
    
    print("\nEvaluating Adversarial Model:")
    preds_proba = clf.predict_proba(X_val)[:, 1]
    preds = clf.predict(X_val)
    
    auc = roc_auc_score(y_val, preds_proba)
    print(f"Adversarial ROC AUC Score: {auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_val, preds, target_names=["Is Train (0)", "Is Test (1)"]))
    
    if auc > 0.8:
        print("\n[CRITICAL FINDING] The ROC AUC is extremely high. The model can easily tell apart the Training rows from the Testing rows. This mathematically proves massive distribution shift/covariate shift between the CSVs. No model trained on the train set will ever generalize to the test set.")
    else:
        print("\n[OK] The datasets are similar. The test set is valid.")

if __name__ == "__main__":
    run_adversarial_validation()
