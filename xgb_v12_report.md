# V12 Deep Robustness & Tuning XGBoost

## Strategy
1. **Missings Native Handling**: Stopped filling NaNs with 0. Allowed the trees to route missing values optimally.
2. **Class Imbalance Scaling**: `scale_pos_weight` mathematically adjusted to target rarity.
3. **Patience & Extreme Tuning**: 10000 boost rounds allowed, early_stopping at 400. Found structural hyperparameters.
    
## Final Performance
- **10-Fold OOF AUC**: **0.91591**

## Top Best Parameters Found
```json
{
  "learning_rate": 0.013009974693589927,
  "max_depth": 4,
  "subsample": 0.9270301648390112,
  "colsample_bytree": 0.3482527518926207,
  "colsample_bylevel": 0.927206652796802,
  "min_child_weight": 22,
  "gamma": 0.22515334099851841,
  "reg_alpha": 2.3806456742660747e-05,
  "reg_lambda": 0.004562251316381654,
  "max_delta_step": 1.4762206881110982
}
```
