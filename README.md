🏎️ F1 Lap Time Predictor with MLOps & SHAP Explainability

This project predicts Formula 1 lap times using telemetry data (FastF1), a Random Forest model, and SHAP explainability inside an interactive Streamlit dashboard.

- Extracts real F1 telemetry from FastF1
- Predicts lap time using:
  - Throttle %
  - DRS %
  - Fuel Load
  - Speed
  - Driver & Team (one-hot encoded)
- Streamlit dashboard:
  - Live prediction UI
  - SHAP explanation plots
  - Feature importances
  - Cross-validation graph
  - Hyperparameter controls (n_estimators, max_depth, etc.)
  - Retrain model button

