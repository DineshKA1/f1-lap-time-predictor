import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib
import os


df = pd.read_csv('data/lap_features.csv')
X = df[['AvgThrottle', 'AvgDRS', 'FuelLoad', 'AvgSpeed']]
y = df['LapTime']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Grid Search for model tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10]
}
grid = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, scoring='neg_mean_absolute_error')
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
joblib.dump(best_model, 'models/model_tuned.pkl')

#Cross-validation predictions 
y_cv_pred = cross_val_predict(best_model, X, y, cv=5)

#Predicted vs Actual LapTime
plt.figure(figsize=(8, 6))
plt.scatter(y, y_cv_pred, color='blue', label='CV Prediction')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Lap Time (s)')
plt.ylabel('Predicted Lap Time (s)')
plt.title('Predicted vs Actual Lap Time (Cross-Validation)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('lap_time_predictions_cv.png')
plt.show()

#Evaluate best model on hold-out test set
y_test_pred = best_model.predict(X_test)
mae = mean_absolute_error(y_test, y_test_pred)
print(f"Best MAE on test set: {mae:.2f} seconds")
