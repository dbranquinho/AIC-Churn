# V13 Target Encoding & ChargeResidual (The Clean Slate) XGBoost

## Strategy
1. **The ChargeResidual Trick**: Replaced noisy features with the exact Kaggle solution metrics: `charge_residual_sign` and `charge_residual_relative`.
2. **Strict In-Fold Target Encoding**: Swapped native `enable_categorical` and OneHot for high-power probabilistic `category_encoders.TargetEncoder(smoothing=12.40)`, computed securely inside the validation segments to prevent data leakage.
3. **Clean Foundation**: Dropped scaling penalties and Original Dataset mixing that skewed bounds. Brought back classic `fillna(0)`.
    
## Final Performance
- **10-Fold OOF AUC**: **0.91673**

## Top Best Parameters Found
```json
{
  "learning_rate": 0.013732699451778858,
  "max_depth": 6,
  "subsample": 0.7899531740717594,
  "colsample_bytree": 0.30633860780417865,
  "min_child_weight": 13,
  "gamma": 0.040019972839629867,
  "reg_alpha": 2.3418160307991083,
  "reg_lambda": 0.0001097432926339284,
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "tree_method": "hist",
  "device": "cuda",
  "random_state": 42,
  "te_smoothing": 12.403326867061793
}
```
