# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Healthcare Predictive Analytics", layout="wide")
st.title("Healthcare Predictive Analytics Dashboard 🏥")

# ------------------------------
# Load Data
# ------------------------------
df = pd.read_csv("clean_diabetes.csv")
df_inflow = pd.read_csv("patient_inflow.csv")

# ------------------------------
# Preprocess for model
# ------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train RandomForest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_scaled, y)

# ------------------------------
# Tabs
# ------------------------------
tabs = st.tabs(["Dataset Overview", "Disease Risk Prediction", "Real-Time Monitoring", "Resource Optimization"])

# -------- Dataset Overview --------
with tabs[0]:
    st.header("Dataset Overview")
    st.dataframe(df.head())
    st.write("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

# -------- Disease Risk Prediction --------
with tabs[1]:
    st.header("Disease Risk Prediction (Random Forest)")
    st.write("Enter patient data to predict diabetes risk:")

    input_data = []
    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].median()))
        input_data.append(val)

    if st.button("Predict"):
        data_scaled = scaler.transform([input_data])
        pred = rf.predict(data_scaled)
        pred_proba = rf.predict_proba(data_scaled)[0][1]

        if pred[0] == 1:
            st.warning(f"⚠️ High risk of diabetes! Probability: {pred_proba:.2f}")
        else:
            st.success(f"✅ Low risk of diabetes. Probability: {pred_proba:.2f}")

# -------- Real-Time Monitoring --------
with tabs[2]:
    st.header("Real-Time Monitoring (Simulated)")
    st.write("Monitoring simulated wearable device data:")

    new_data_stream = [
        [2, 120, 70, 22, 80, 28.5, 0.35, 32],
        [5, 150, 85, 33, 150, 31.2, 0.5, 40],
        [1, 85, 66, 29, 0, 26.6, 0.351, 31],
    ]

    for i, data in enumerate(new_data_stream):
        data_scaled = scaler.transform([data])
        pred = rf.predict(data_scaled)
        if pred[0] == 1:
            st.warning(f"Data point {i+1}: ⚠️ High risk of diabetes!")
        else:
            st.success(f"Data point {i+1}: ✅ Condition stable")

# -------- Resource Optimization --------
with tabs[3]:
    st.header("Resource Optimization")
    st.write("Forecasting patient inflow and required resources:")

    df_inflow['forecast_next_day'] = df_inflow['patients'].rolling(window=7).mean().shift(1)
    df_inflow['forecast_next_day'] = df_inflow['forecast_next_day'].fillna(method='bfill')
    df_inflow['beds_needed'] = df_inflow['forecast_next_day'].apply(lambda x: int(round(x)))
    df_inflow['doctors_needed'] = df_inflow['forecast_next_day'].apply(lambda x: int(round(x/5)))

    st.dataframe(df_inflow.tail(10))

    fig2 = px.line(df_inflow, x='day', y='patients', title="Daily Patient Inflow")
    st.plotly_chart(fig2)
