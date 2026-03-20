# V11 The Golden Trick Report (Original Dataset Injection)

## Strategy
1. **The Ultimate Cure**: Overfitting synthetic data is solved by concatenating the **Originial IBM Telco Churn Dataset** seamlessly alongside the generated training folds. 
2. **Feature Alchemy & Stacking**: Maintained V9 and V10 techniques: Logistic Meta-Model mapping the outputs of 3 robust Gradient Boosters across 20-trials Optuna.
    
## Final Performance
- **Combined Train + Orig OOF XGBoost AUC**: 0.91576
- **Combined Train + Orig OOF LightGBM AUC**: 0.91554
- **Combined Train + Orig OOF CatBoost AUC**: 0.91549
- **Meta-Model Learned Weights**: XGB: 2.131, LGB: 2.132, CAT: 2.110
- **Final V11 Golden Trick Stacked OOF AUC**: **0.91597**
