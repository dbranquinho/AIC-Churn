# V8 Pure Robust XGBoost Report

## Strategy
1. **Model Architecture**: Reverted to a single, highly powerful XGBoost architecture to reduce dilution.
2. **Robust Hyperparameters**: 
   - 100 Optuna trials
   - Expanded regularization boundaries (`colsample_bylevel`, `max_delta_step`, tighter `learning_rate` floor).
3. **Financial Features**: Included V3 top performers like `avg_charge_per_tenure` and discrepency ratios.
    
## Final Performance
- **10-Fold OOF AUC**: **0.91675**

## Top Best Parameters Found
```json
{
  "learning_rate": 0.02797309761297987,
  "max_depth": 7,
  "subsample": 0.9495759350842995,
  "colsample_bytree": 0.3717520047951345,
  "colsample_bylevel": 0.855444225341255,
  "min_child_weight": 18,
  "max_delta_step": 6.7618241945843165,
  "gamma": 0.2815502235323768,
  "reg_alpha": 10.03870883709019,
  "reg_lambda": 7.047968521508006
}
```
