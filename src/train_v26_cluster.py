import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Normalizer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings('ignore')

try:
    import umap
except ImportError:
    print("UMAP not installed. Please run: pip install umap-learn")
    sys.exit(1)

# Ensure correct path to import src.config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')
os.makedirs(SUBMISSION_DIR, exist_ok=True)

def main():
    print("="*70)
    print(" V26: DISRUPTIVE CLUSTERING APPROACH (Cosine Similarity) ")
    print("="*70)
    
    # Load data
    train_df = pd.read_csv(config.KAGGLE_TRAIN_PATH)
    test_df = pd.read_csv(config.KAGGLE_TEST_PATH)
    
    # Handle target
    y = train_df[config.KAGGLE_TARGET_COL].map({'Yes': 1, 'No': 0, 'True': 1, 'False': 0, '1': 1, '0': 0}).fillna(0).astype('int64').values
    train_df = train_df.drop(columns=[config.KAGGLE_TARGET_COL, config.KAGGLE_ID_COL], errors='ignore')
    
    test_ids = test_df[config.KAGGLE_ID_COL].values
    test_df = test_df.drop(columns=[config.KAGGLE_ID_COL], errors='ignore')
    
    # Save original data for profiling
    orig_train = train_df.copy()
    
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
        test_df[col] = pd.to_numeric(test_df[col], errors='coerce').fillna(0)
        
    cat_cols = [c for c in train_df.columns if c not in numeric_cols]
    for col in cat_cols:
        train_df[col] = train_df[col].fillna('Missing').astype(str)
        test_df[col] = test_df[col].fillna('Missing').astype(str)
        
    print("\nFeature Engineering: One-Hot Encoding and Scaling...")
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
        ]
    )
    
    X_train_processed = preprocessor.fit_transform(train_df)
    X_test_processed = preprocessor.transform(test_df)
    
    print(f"Processed Train Shape: {X_train_processed.shape}")
    
    print("Applying L2 Normalization for Cosine Similarity modeling...")
    normalizer = Normalizer(norm='l2')
    X_train_norm = normalizer.fit_transform(X_train_processed)
    X_test_norm = normalizer.transform(X_test_processed)
    
    print("\nSearching for optimal number of clusters (Silhouette Score target > 0.75)...")
    sample_size = min(10000, X_train_norm.shape[0])
    np.random.seed(42)
    idx_sample = np.random.choice(X_train_norm.shape[0], sample_size, replace=False)
    X_sample = X_train_norm[idx_sample]
    
    best_k = -1
    best_score = -1
    best_model = None
    
    for k in range(2, 21):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels_sample = kmeans.fit_predict(X_sample)
        
        score = silhouette_score(X_sample, cluster_labels_sample, metric='cosine')
        print(f" K={k:02d} -> Silhouette Score (Cosine): {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = kmeans
            
        if score > 0.75:
            print(f"--> Reached target score > 0.75 with K={k}!")
            break
            
    if best_score <= 0.75:
        print(f"--> Could not reach 0.75. Using best K found: K={best_k} with score {best_score:.4f}")
        
    print(f"\nFitting final clustering model with K={best_k} on full train data...")
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    train_clusters = final_kmeans.fit_predict(X_train_norm)
    
    # -------------------------------------------------------------
    # Profiling
    # -------------------------------------------------------------
    print("\n" + "="*50)
    print(" CLUSTER PROFILING ")
    print("="*50)
    
    profile_df = orig_train.copy()
    for col in numeric_cols:
        profile_df[col] = pd.to_numeric(profile_df[col], errors='coerce').fillna(0)
    profile_df['Cluster'] = train_clusters
    profile_df['Churn'] = y
    
    cluster_stats = []
    
    for c in range(best_k):
        c_mask = profile_df['Cluster'] == c
        c_size = c_mask.sum()
        c_churn_rate = profile_df.loc[c_mask, 'Churn'].mean()
        
        stats = {'Cluster': c, 'Size': c_size, 'ChurnRate': c_churn_rate}
        
        for col in numeric_cols:
            stats[f'{col}_mean'] = profile_df.loc[c_mask, col].mean()
            
        for col in cat_cols:
            if not profile_df.loc[c_mask, col].empty:
                stats[f'{col}_mode'] = profile_df.loc[c_mask, col].mode()[0]
            else:
                stats[f'{col}_mode'] = 'N/A'
                
        cluster_stats.append(stats)
        
    stats_df = pd.DataFrame(cluster_stats)
    print(stats_df[['Cluster', 'Size', 'ChurnRate'] + [f'{col}_mean' for col in numeric_cols]].to_string(index=False))
    
    print("\nVisualizing high/low churn clusters...")
    for index, row in stats_df.iterrows():
        print(f"Cluster {row['Cluster']}: Size={row['Size']}, ChurnRate={row['ChurnRate']:.2%} => "
              f"TenureMean={row['tenure_mean']:.1f}, "
              f"Distribution across key categorical mode: {row.get('Contract_mode', 'N/A')}")
              
    # -------------------------------------------------------------
    # UMAP Visualization
    # -------------------------------------------------------------
    print("\nGenerating UMAP 2D Projection (metric='cosine')...")
    
    umap_sample_size = min(15000, X_train_norm.shape[0])
    idx_umap = np.random.choice(X_train_norm.shape[0], umap_sample_size, replace=False)
    X_umap_sample = X_train_norm[idx_umap]
    y_umap_sample = y[idx_umap]
    c_umap_sample = train_clusters[idx_umap]
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)
    embedding = reducer.fit_transform(X_umap_sample)
    
    plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 2, 1)
    scatter1 = plt.scatter(embedding[:, 0], embedding[:, 1], c=c_umap_sample, cmap='tab20', s=5, alpha=0.7)
    plt.title('UMAP Projection by Cluster')
    plt.colorbar(scatter1, label='Cluster ID')
    
    plt.subplot(1, 2, 2)
    scatter2 = plt.scatter(embedding[:, 0], embedding[:, 1], c=y_umap_sample, cmap='coolwarm', s=5, alpha=0.7)
    plt.title('UMAP Projection by True Churn Label')
    plt.colorbar(scatter2, label='Churn (0=No, 1=Yes)')
    
    plt.tight_layout()
    plot_path = os.path.join(config.BASE_DIR, 'umap_clusters_v26.png')
    plt.savefig(plot_path)
    print(f"UMAP visualization saved to {plot_path}")
    
    # -------------------------------------------------------------
    # Prediction on Test Set & Submission
    # -------------------------------------------------------------
    print("\nAssigning Test Set to Clusters...")
    test_clusters = final_kmeans.predict(X_test_norm)
    
    cluster_churn_map = stats_df.set_index('Cluster')['ChurnRate'].to_dict()
    global_churn_mean = y.mean()
    
    test_preds = np.array([cluster_churn_map.get(c, global_churn_mean) for c in test_clusters])
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: test_ids,
        config.KAGGLE_TARGET_COL: test_preds
    })
    
    sub_path = os.path.join(SUBMISSION_DIR, 'submission_v26_cluster.csv')
    sub_df.to_csv(sub_path, index=False)
    print(f"\nFinal V26 (Clustering) saved to: {sub_path}")

if __name__ == "__main__":
    main()
