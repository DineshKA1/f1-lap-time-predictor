import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("data/lap_features.csv")

#One-hot encode Driver and Team
df_encoded = pd.get_dummies(df, columns=["Driver", "Team"])

X = df_encoded.drop("LapTime", axis=1)
y = df_encoded["LapTime"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Save model and feature columns
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model_tuned.pkl")
joblib.dump(X.columns.tolist(), "models/feature_columns.pkl")

print("Model and feature columns saved.")
