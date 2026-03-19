# V6 Deep Tuning Ensemble (Back to Basics) Report

## Strategy
1. **Clean Features**: Dropped completely all financial ratio math, forcing trees to evaluate purely raw signals without confusion.
2. **Deep Tuning**: Unleashed a dedicated 30-trial Optuna study specifically matching hyperparameters for each model independently.
    
## Final Performance
- **Tuned XGBoost AUC**: 0.91652
- **Tuned LightGBM AUC**: 0.91651
- **Tuned CatBoost AUC**: 0.91625
- **Final Blended Ensemble OOF AUC**: **0.91678**
