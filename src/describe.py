import pandas as pd
import src.config as config

print("Loading Data...")
df_train = pd.read_csv(config.TRAIN_DATA_PATH).dropna()
df_test = pd.read_csv(config.TEST_DATA_PATH).dropna()

print("\n--- Training Data Description ---")
print(df_train.describe())

print("\n--- Testing Data Description ---")
print(df_test.describe())
