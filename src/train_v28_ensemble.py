import os
import sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import src.config as config

SUBMISSION_DIR = os.path.join(config.BASE_DIR, 'submissions')

def main():
    print("="*70)
    print(" V28: THE FINAL ENSEMBLE (GBDT + RealMLP Neural Network) ")
    print("="*70)
    
    # Define paths
    tree_sub_path = os.path.join(SUBMISSION_DIR, 'submission_v25_v6_revival.csv')
    nn_sub_path = os.path.join(SUBMISSION_DIR, 'submission_v26_realmlp.csv')
    
    if not os.path.exists(tree_sub_path):
        print(f"ERRO: A submissão das Árvores (V25) não foi encontrada em: {tree_sub_path}")
        print("Por favor, garanta que você já rodou o train_v25_v6_revival.py recentemente.")
        sys.exit(1)
        
    if not os.path.exists(nn_sub_path):
        print(f"ERRO: A submissão da Rede Neural (V26) não foi encontrada em: {nn_sub_path}")
        print("Por favor, garanta que você já rodou o train_v26_realmlp.py.")
        sys.exit(1)
        
    print(f"Carregando Predições do Tri-Ensemble de Árvores (V25)...")
    tree_df = pd.read_csv(tree_sub_path)
    
    print(f"Carregando Predições da Rede Neural RealMLP (V26)...")
    nn_df = pd.read_csv(nn_sub_path)
    
    # Ensure IDs match perfectly
    if not (tree_df[config.KAGGLE_ID_COL] == nn_df[config.KAGGLE_ID_COL]).all():
        print("ERRO CRÍTICO: Os IDs (Client IDs) das duas submissões não batem. Impossível mesclar.")
        sys.exit(1)
        
    # Kaggle Grandmster Blending Logic
    # Dado que Árvores (~0.917) são significativamente mais fortes que NN (~0.913) neste dataset,
    # as Redes Neurais não devem ter peso total. Elas agem apenas "anulando" bordas muito duras (overfitting) do XGBoost.
    # Pesos clássicos para essa heterogeneidade: 85/15 ou 90/10.
    
    weights = [
        (0.95, 0.05),
        (0.90, 0.10),
        (0.85, 0.15),
        (0.80, 0.20)
    ]
    
    print("\nGerando Multi-Submissões de Blending...")
    for w_tree, w_nn in weights:
        blend_preds = (tree_df[config.KAGGLE_TARGET_COL] * w_tree) + (nn_df[config.KAGGLE_TARGET_COL] * w_nn)
        
        blend_df = pd.DataFrame({
            config.KAGGLE_ID_COL: tree_df[config.KAGGLE_ID_COL],
            config.KAGGLE_TARGET_COL: blend_preds
        })
        
        # Name format: submission_v28_blend_85_15.csv
        blend_name = f'submission_v28_blend_{int(w_tree*100):02d}_{int(w_nn*100):02d}.csv'
        out_path = os.path.join(SUBMISSION_DIR, blend_name)
        
        blend_df.to_csv(out_path, index=False)
        print(f" -> Salvo: {blend_name} (Árvores: {w_tree*100:02.0f}% | PyTorch: {w_nn*100:02.0f}%)")
        
    print("\n" + "="*70)
    print(" ENSEMBLE CONCLUÍDO COM SUCESSO ")
    print("="*70)
    print("Instrução Final: Vá no Kaggle e submeta PRIMEIRO o arquivo 'submission_v28_blend_85_15.csv'.")
    print("Se ele aumentar seu score, teste o '90_10' para calibrar a influência do PyTorch.")

if __name__ == "__main__":
    main()
