import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# App title
st.set_page_config(page_title="SCM Demand Forecasting", layout="wide")
st.title("📈 Supply Chain Demand Forecasting Dashboard")

# Load model and scaler
MODEL_PATH = os.path.join("models", "demand_forecasting_model.keras")
SCALER_PATH = os.path.join("models", "scaler.pkl")
FEATURE_COLS_PATH = os.path.join("models", "feature_columns.pkl")

@st.cache_resource
def load_artifacts():
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_columns = joblib.load(FEATURE_COLS_PATH)
    return model, scaler, feature_columns

model, scaler, feature_columns = load_artifacts()

# Sidebar: file uploader
st.sidebar.header("Upload Data")
uploaded_file = st.sidebar.file_uploader("Choose CSV file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("File uploaded successfully!")

    # Preprocessing
    st.header("Raw Data Preview")
    st.dataframe(data.head())

    if 'Date' in data.columns:
        data['Date'] = pd.to_datetime(data['Date'])
        data['Month'] = data['Date'].dt.month
        data['DayOfWeek'] = data['Date'].dt.dayofweek
        data['Quarter'] = data['Date'].dt.quarter
        data = data.drop(columns=['Date'])

    # One-hot encoding
    cat_cols = data.select_dtypes(include='object').columns
    data_enc = pd.get_dummies(data, columns=cat_cols, drop_first=True)

    # Align columns
    data_enc = data_enc.reindex(columns=feature_columns, fill_value=0)
    X = data_enc.values

    # Scale and predict
    X_scaled = scaler.transform(X)
    preds = model.predict(X_scaled).flatten()
    data['Predicted Sales'] = preds

    # Show predictions
    st.header("Predictions Preview")
    st.dataframe(data[['Predicted Sales']].head())

    # Download button
    st.download_button(
        label="📥 Download Predictions",
        data=data.to_csv(index=False),
        file_name="predicted_sales.csv",
        mime="text/csv"
    )

    # Bar chart preview
    st.subheader("📊 Predicted Sales Chart")
    st.bar_chart(data[['Predicted Sales']].head(20))

    # Score metrics if target column is present
    target_col = 'Number of products sold'
    if target_col in data.columns:
        mse = mean_squared_error(data[target_col], preds)
        r2 = r2_score(data[target_col], preds)
        st.metric("Mean Squared Error", f"{mse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")

        # Scatter plot
        fig, ax = plt.subplots()
        ax.scatter(data[target_col], preds, alpha=0.6)
        ax.set_xlabel('True Sales')
        ax.set_ylabel('Predicted Sales')
        ax.set_title('True vs Predicted Sales')
        st.pyplot(fig)

else:
    st.info("Upload a CSV file to get started.")

#                   "C:\\Users\\mdana\\AppData\\Local\\Programs\\Python\\Python310\\python.exe" -m venv .venv310
#                .venv310\\Scripts\\activate.bat
#              streamlit run streamlit_app/app.py

