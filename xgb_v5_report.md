# V5 Tri-Ensemble (No One-Hot) Report
    
## Strategy:
1. **Zero One-Hot Encoding**: Handed off all categorical features strictly as `dtype='category'` to XGBoost / LGBM / CatBoost.
2. **Tri-Ensemble**: Combined 3 elite gradient boosting libraries.
3. **10-Fold CV**: Reduced variance significantly over 5-fold CV.

## Final Performance
- **XGBoost AUC**: 0.91665
- **LightGBM AUC**: 0.91572
- **CatBoost AUC**: 0.91557
- **Final Blended Ensemble OOF AUC**: **0.91626**
