import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
import shap
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Set Streamlit page config
st.set_page_config(page_title="F1 Lap Time Predictor", layout="wide")


DATA_PATH = "data/lap_features.csv"
MODEL_PATH = "models/model_tuned.pkl"
FEATURE_COLS_PATH = "models/feature_columns.pkl"

# Load data and model
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    return None

@st.cache_data
def load_feature_columns():
    if os.path.exists(FEATURE_COLS_PATH):
        return joblib.load(FEATURE_COLS_PATH)
    return []

# Preprocessing function
def preprocess_input(input_row, feature_columns):
    df_input = pd.DataFrame([input_row])
    df_encoded = pd.get_dummies(df_input)

    # Add any missing columns with 0
    for col in feature_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0

    return df_encoded[feature_columns]


df = load_data()
model = load_model()
feature_columns = load_feature_columns()

# One-hot encode the full dataset
X = pd.get_dummies(df[['AvgThrottle', 'AvgDRS', 'FuelLoad', 'AvgSpeed', 'Driver', 'Team']])
y = df['LapTime']

# UI: Sidebar Inputs
st.sidebar.header("📥 Input Lap Features")

avg_throttle = st.sidebar.slider("Average Throttle (%)", 0.0, 100.0, 80.0)
avg_drs = st.sidebar.slider("Average DRS Activation (%)", 0.0, 100.0, 60.0)
fuel_load = st.sidebar.slider("Fuel Load (kg)", 0.0, 100.0, 50.0)
avg_speed = st.sidebar.slider("Average Speed (km/h)", 100.0, 350.0, 250.0)

unique_drivers = df["Driver"].unique()
unique_teams = df["Team"].unique()

driver = st.sidebar.selectbox("Driver", sorted(unique_drivers))
team = st.sidebar.selectbox("Team", sorted(unique_teams))

input_row = {
    "AvgThrottle": avg_throttle,
    "AvgDRS": avg_drs,
    "FuelLoad": fuel_load,
    "AvgSpeed": avg_speed,
    "Driver": driver,
    "Team": team
}

input_df = preprocess_input(input_row, feature_columns)

st.title("🏎️ F1 Lap Time Predictor + Explainability")

#Hyperparameter UI for retraining
st.sidebar.subheader("Retrain Hyperparameters")

n_estimators = st.sidebar.slider("Number of Trees", 10, 500, 100, step=10)
max_depth = st.sidebar.selectbox("Maximum Depth", [None, 5, 10, 20], index=0)
min_samples_split = st.sidebar.slider("Min Samples Split", 2, 10, 2)

#Retraining block
if st.button("Retrain Model"):
    with st.spinner("Training new model"):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)

        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        joblib.dump(X.columns.tolist(), FEATURE_COLS_PATH)

        st.success("Model retrained and saved.")
        model = load_model()  
        st.rerun()

#Prediction and explanation block
if model:
    if st.button("⚡ Predict Lap Time"):
        prediction = model.predict(input_df)[0]
        st.success(f"Predicted Lap Time: **{prediction:.2f} seconds**")

        X_shap = X.astype(float)
        input_df_shap = input_df.astype(float)

        explainer = shap.Explainer(model.predict, X_shap)
        shap_values = explainer(input_df_shap)

        #SHAP Explanation
        st.subheader("SHAP Explanation")
        try:
            X_shap = X.astype(float)
            input_df_shap = input_df.astype(float)

            explainer = shap.Explainer(model.predict, X_shap)
            shap_values = explainer(input_df_shap)

            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")


    #Feature importance chart
    with st.expander("Feature Importances"):
        importances = pd.Series(model.feature_importances_, index=feature_columns).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        importances.plot(kind='barh', ax=ax)
        ax.set_title("Feature Importances")
        st.pyplot(fig)

    #Input preview
    with st.expander("Input Data"):
        st.write(input_df)

    #Static CV plot if exists
    if os.path.exists("lap_time_predictions_cv.png"):
        st.image("lap_time_predictions_cv.png", caption="Predicted vs Actual (CV)", use_column_width=True)
else:
    st.warning("No model found. Please retrain first.")
