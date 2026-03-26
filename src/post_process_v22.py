"""
V22 - Post-Processing Magic: Rank Power Averaging
When training new models hits a strict mathematical plateau, Grandmasters turn to Ensembling.
This script does NOT average the raw probabilities (which causes signal dilution due to different model calibrations).
Instead, it:
1. Converts each model's predictions into strict Percentile Ranks.
2. Raises the rank to a Power (e.g., 2.5) to exponentially reward high-confidence signals.
3. Smooths out the decision boundary, boosting the Public LB score immediately without training anything.
"""
import pandas as pd
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import rankdata

import src.config as config

SUBMISSIONS_DIR = os.path.join(config.BASE_DIR, 'submissions')
OUTPUT_PATH = os.path.join(SUBMISSIONS_DIR, 'submission_v22_power_blend.csv')

# --------------------------------------------------------------------------------
# CONFIG: Coloque aqui o nome EXATO dos arquivos CSV que te deram a maior nota no Kaggle.
# Misturamos modelos de base muito diferentes para maximizar o ganho geométrico.
# --------------------------------------------------------------------------------
BEST_SUBMISSIONS = [
    'submission_v29_chris_deotte_exact.csv', # Novo Recorde Absoluto (0.91447)
    'submission_v6_ensemble.csv',            # Seu antigo recorde purista (0.91416)
    'submission_v23_autogluon.csv'           # A diversidade máxima de arquiteturas (AutoGluon)
]

def power_average(submissions_list, power=2.5):
    print(f"Applying Rank Power Averaging (Power={power}) on {len(submissions_list)} submissions...")
    
    base_df = pd.read_csv(os.path.join(SUBMISSIONS_DIR, submissions_list[0]))
    ids = base_df['id'] if 'id' in base_df.columns else base_df[config.KAGGLE_ID_COL]
    
    blended_rank = np.zeros(len(base_df))
    valid_count = 0
    
    for sub in submissions_list:
        path = os.path.join(SUBMISSIONS_DIR, sub)
        if not os.path.exists(path):
            print(f"  [ERROR] File not found: {sub}. Skipping.")
            continue
            
        df = pd.read_csv(path)
        
        # 1. Transform raw probabilities into Uniform Percentile Ranks [0, 1]
        ranks = rankdata(df[config.KAGGLE_TARGET_COL]) / len(df)
        
        # 2. Exponentiate to "stretch" high-confidence vs low-confidence edges
        power_rank = ranks ** power
        blended_rank += power_rank
        valid_count += 1
        
        print(f"  [SUCCESS] Processed -> {sub}")
        
    if valid_count == 0:
        print("No valid submission files processed. Exiting.")
        return
        
    # 3. Average the weighted ranks back
    blended_rank = blended_rank / valid_count
    
    sub_df = pd.DataFrame({
        config.KAGGLE_ID_COL: ids,
        config.KAGGLE_TARGET_COL: blended_rank
    })
    
    sub_df.to_csv(OUTPUT_PATH, index=False)
    print("\n" + "="*70)
    print(f"  DONE! Saved Ultimate Magic Blend to:\n  -> {OUTPUT_PATH}")
    print("="*70)
    print("  Submeta ESTE arquivo V22 no Kaggle. Ele funde a generalização de todo o seu histórico!")

if __name__ == "__main__":
    power_average(BEST_SUBMISSIONS, power=2.5)
