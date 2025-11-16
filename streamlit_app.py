import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ------------------------------------------
# Load Data
# ------------------------------------------
df = pd.read_csv("/kaggle/input/pima-indians-diabetes-database/diabetes.csv")

# Replace zero values with median
cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

# ------------------------------------------
# ML MODEL TRAINING
# ------------------------------------------
X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Models
log_model = LogisticRegression()
log_model.fit(X_train, y_train)

dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Choose best model
best_model = rf_model

# ------------------------------------------
# Streamlit Dashboard
# ------------------------------------------
st.title("📊 Healthcare Predictive Analytics Dashboard")
st.write("AI-powered dashboard for patient care enhancement")

tabs = st.tabs(["📁 Data Overview", "🤖 Disease Prediction", "📡 Real-time Monitoring", "🏥 Resource Optimization"])


# ---------------------------------------------------
# TAB 1 — DATA OVERVIEW
# ---------------------------------------------------
with tabs[0]:
    st.header("🔍 Data Overview")
    st.write(df.head())

    st.subheader("📌 Statistical Summary")
    st.write(df.describe())

    st.subheader("📉 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)


# ---------------------------------------------------
# TAB 2 — DISEASE PREDICTION
# ---------------------------------------------------
with tabs[1]:
    st.header("🤖 Disease Risk Prediction")

    st.write("Enter patient values below to predict diabetes risk:")

    input_data = []
    cols_input = X.columns

    col1, col2, col3 = st.columns(3)

    for i, col in enumerate(cols_input):
        if i % 3 == 0:
            value = col1.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        elif i % 3 == 1:
            value = col2.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        else:
            value = col3.number_input(col, float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        input_data.append(value)

    if st.button("🔮 Predict"):
        scaled = scaler.transform([input_data])
        result = best_model.predict(scaled)[0]

        if result == 1:
            st.error("⚠️ High risk of diabetes detected!")
        else:
            st.success("✅ No diabetes risk detected.")


# ---------------------------------------------------
# TAB 3 — REAL-TIME MONITORING
# ---------------------------------------------------
with tabs[2]:
    st.header("📡 Real-time Monitoring Simulation")

    new_data_stream = [
        [2, 120, 70, 22, 80, 28.5, 0.35, 32],
        [5, 150, 85, 33, 150, 31.2, 0.5, 40],
        [1, 85, 66, 29, 100, 26.6, 0.351, 31],
    ]

    if st.button("▶️ Start Real-Time Simulation"):
        for data in new_data_stream:
            data_scaled = scaler.transform([data])
            pred = best_model.predict(data_scaled)[0]

            if pred == 1:
                st.error("⚠️ Alert: High diabetes risk!")
            else:
                st.success("✅ Stable condition.")

            time.sleep(1)


# ---------------------------------------------------
# TAB 4 — RESOURCE OPTIMIZATION
# ---------------------------------------------------
with tabs[3]:
    st.header("🏥 Resource Optimization (Forecasting)")

    # Simulated inflow data
    df_inflow = pd.DataFrame({
        'day': pd.date_range(start='2025-10-01', periods=30, freq='D'),
        'predicted_patients': [25, 30, 28, 32, 27, 26, 29, 31, 30, 28,
                               26, 27, 30, 32, 33, 31, 29, 28, 30, 27,
                               29, 31, 32, 30, 28, 29, 30, 31, 32, 33]
    })

    df_inflow['forecast_next_day'] = df_inflow['predicted_patients'].rolling(window=7).mean().shift(1)
    df_inflow['forecast_next_day'] = df_inflow['forecast_next_day'].fillna(method='bfill')

    df_inflow['doctors_needed'] = (df_inflow['forecast_next_day'] / 5).round().astype(int)
    df_inflow['beds_needed'] = df_inflow['forecast_next_day'].round().astype(int)

    st.write(df_inflow.tail(10))

    # Plot forecast
    fig2, ax2 = plt.subplots(figsize=(10,5))
    ax2.plot(df_inflow['day'], df_inflow['predicted_patients'], label='Actual Patients')
    ax2.plot(df_inflow['day'], df_inflow['forecast_next_day'], label='Forecast')
    ax2.legend()
    st.pyplot(fig2)
