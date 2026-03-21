# V14 Pseudo-Labeling & Semi-Supervised Learning XGBoost

## Strategy
1. **The Overfitting Panacea**: Reached the mathematical limit of the training data. Shifted to Semi-Supervised Pseudo-Labeling.
2. **Confidence Injection**: Read the `submission_v13` probabilities. **Injected 87643 test records (Pos: 2, Neg: 87641)** with >98.5% and <1.5% confidence were given fake `Churn` targets and appended securely back to `train.csv`.
3. **Target Encoding**: Retained `TargetEncoder` and `ChargeResidual`.
    
## Final Performance
- **10-Fold OOF AUC (Train + Pseudo)**: **0.92968**

## Top Best Parameters Found
```json
{
  "learning_rate": 0.03229243963427607,
  "max_depth": 6,
  "subsample": 0.8501376696196967,
  "colsample_bytree": 0.3016242963770833,
  "min_child_weight": 18,
  "gamma": 0.003627484233742829,
  "reg_alpha": 0.1135966159982872,
  "reg_lambda": 12.095855865207273,
  "objective": "binary:logistic",
  "eval_metric": "auc",
  "tree_method": "hist",
  "device": "cuda",
  "random_state": 42,
  "te_smoothing": 37.19463659026127
}
```
