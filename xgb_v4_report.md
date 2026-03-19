# XGBoost v4 (CatBoostEncoder Edition) Report

## Strategy & Extreme Features
This script completely drops OneHot / Native categorical support to shatter the 0.91757 ceiling:
1. **CatBoost Encoder**: Deployed `category_encoders.CatBoostEncoder` entirely out-of-fold to prevent leakage while providing strictly regularized numerical assignments for all categories.
2. **Financial Ratios & Profiles**: KMeans clustering ($k=8$), percentile binning, cross-ratios.
3. **Deep Optuna Optimization (80 Trials)**: Full continuous dataset processing.
4. **Ensembling**: 5-Fold Stratified Average.

## Final Performance
- **Out-Of-Fold (OOF) ROC-AUC**: **0.91558**

## Top 15 Most Important Features
| Feature | Importance Score (Ensemble Sum) |
|---------|--------------------------------|
| TotalCharges | 25995.0 |
| MonthlyCharges | 16921.0 |
| avg_charge_per_tenure | 16330.0 |
| tenure_over_monthly | 14199.0 |
| pct_discrepancy | 13205.0 |
| monthly_over_total | 12473.0 |
| charge_discrepancy | 12454.0 |
| expected_total_charges | 10839.0 |
| tenure | 8934.0 |
| Contract_Payment_cb_enc | 4685.0 |
| total_services | 2709.0 |
| PaymentMethod_cb_enc | 2685.0 |
| financial_cluster_cb_enc | 2601.0 |
| MultipleLines_cb_enc | 2505.0 |
| Contract_Payment_freq | 2458.0 |
