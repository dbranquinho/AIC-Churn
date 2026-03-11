# XGBoost v3 (Target-Breaker) Report

## Strategy & Extreme Features
This script was rewritten aggressively to break the 0.91752 performance barrier:
1. **Financial Ratios & Profiles**: Added KMeans clustering ($k=8$), percentile binning, and cross-ratios (`Monthly / Total`).
2. **Smoothed Target Encoding**: OOF encoding is now smoothed and Gaussian-noised to prevent high-cardinality leakage.
3. **Deep Optuna Optimization (80 Trials)**: Bound learning rate to `[0.005, 0.05]` to force slow, highly accurate tree growth.
4. **Ensembling**: A robust 5-Fold Stratified Average.

## Final Performance
- **Out-Of-Fold (OOF) ROC-AUC**: **0.91648**

## Top 15 Most Important Features
| Feature | Importance Score (Ensemble Sum) |
|---------|--------------------------------|
| TotalCharges | 19344.0 |
| MonthlyCharges | 12030.0 |
| avg_charge_per_tenure | 11460.0 |
| Contract_target_enc | 11196.0 |
| PaymentMethod_target_enc | 10975.0 |
| InternetService_target_enc | 10851.0 |
| Contract_Payment_target_enc | 10667.0 |
| tenure_over_monthly | 9820.0 |
| financial_cluster_target_enc | 9733.0 |
| pct_discrepancy | 9530.0 |
| charge_discrepancy | 8454.0 |
| monthly_over_total | 7861.0 |
| expected_total_charges | 6973.0 |
| tenure | 6586.0 |
| Contract_Payment | 5902.0 |
