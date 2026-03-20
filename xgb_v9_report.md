# V9 Feature Alchemy XGBoost Report

## Strategy
1. **Model Architecture**: Retained pure XGBoost pipeline.
2. **Feature Alchemy**: 
   - `num_services` and `monthly_per_service` created to map structural complexity.
   - Demographics combinations (`is_family`, `is_vulnerable`) passed as binary keys.
   - **Soft KMeans Distances**: A complete space mapping generating 5 orthogonal metric columns.
3. **Robust Hyperparameters**: 50 thorough Optuna trials.
    
## Final Performance
- **10-Fold OOF AUC**: **0.91662**

## Top Best Parameters Found
```json
{
  "learning_rate": 0.030626851977722632,
  "max_depth": 6,
  "subsample": 0.9338965896669214,
  "colsample_bytree": 0.30547960480932823,
  "colsample_bylevel": 0.8192871461681143,
  "min_child_weight": 16,
  "max_delta_step": 7.096651545398151,
  "gamma": 1.0472869367143085,
  "reg_alpha": 0.007555717097453295,
  "reg_lambda": 0.022707640364933925
}
```
