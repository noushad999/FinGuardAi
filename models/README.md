# models/

Deployable model bundles are written here by `python -m ml train-all` (and materialised from the
MLflow registry by `python -m ml deploy`). They are not committed: run `make pipeline` or
`make pipeline-local` to create them.

```
fraud_xgb/            XGBoost P(fraud) + Platt calibrator + feature reference + SHAP importance
ato_xgb/              XGBoost P(account takeover)
fraud_iforest/        Isolation Forest + score percentiles
behavioral_anomaly/   customer-baseline anomaly reference distributions
churn_xgb/            churn model (30-day horizon)
segmentation/         scaler + K-Means + segment names
```
Every bundle has a `metadata.json`: version, feature schema, parameters, metrics, thresholds,
dataset version and MLflow run id.
