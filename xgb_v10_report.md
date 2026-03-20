# V10 Ultimate Stacking Ensemble Report

## Strategy
1. **The Plateau realization**: Feature Alchemy alone (V9) couldn't break the mathematical constraints of single-model learning in this dataset.
2. **Meta-Model Stacking**: We reintroduced LightGBM and CatBoost, but instead of forcing a simple average, we fed their Out-Of-Fold predictions into a **Logistic Regression Meta-Model**.
3. This creates a Level-2 algorithm that dynamically learns exactly *where* and *when* XGBoost makes a mistake but CatBoost gets it right, balancing the predictions perfectly.
    
## Final Performance
- **OOF XGBoost AUC**: 0.91642
- **OOF LightGBM AUC**: 0.91638
- **OOF CatBoost AUC**: 0.91609
- **Meta-Model Learned Weights**: XGB: 2.140, LGB: 2.140, CAT: 2.110
- **Final V10 Stacked OOF AUC**: **0.91662**
