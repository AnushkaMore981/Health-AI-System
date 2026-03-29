# ==========================================
# AI Health Monitoring System (Advanced)
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import pyttsx3

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------
# Voice Engine
# ---------------------------
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# ---------------------------
# Dataset
# ---------------------------
np.random.seed(42)

data = pd.DataFrame({
    "age": np.random.randint(18, 60, 300),
    "steps": np.random.randint(1000, 15000, 300),
    "sleep_hours": np.random.uniform(4, 9, 300),
    "water_intake": np.random.uniform(1, 4, 300),
    "heart_rate": np.random.randint(60, 100, 300)
})

data["health_score"] = (
    0.3 * (data["steps"] / 15000) +
    0.2 * (data["sleep_hours"] / 8) +
    0.2 * (data["water_intake"] / 3) -
    0.2 * (data["heart_rate"] / 100) +
    0.1 * (1 - data["age"] / 60)
) * 100

X = data.drop("health_score", axis=1)
y = data["health_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------------------
# Models
# ---------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(n_estimators=100)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = {
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds),
        "model": model
    }

best_model_name = max(results, key=lambda x: results[x]["R2"])
best_model = results[best_model_name]["model"]

# ---------------------------
# UI Setup
# ---------------------------
st.set_page_config(page_title="Health AI System", layout="centered")

# ---------------------------
# Sidebar Emergency
# ---------------------------
st.sidebar.title("🚑 Emergency")

if st.sidebar.button("Call Ambulance"):
    st.sidebar.error("📞 Call 108 Immediately!")
    st.sidebar.markdown("[📱 Click to Call](tel:108)")

st.sidebar.markdown("🏥 [Find Nearby Hospitals](https://www.google.com/maps/search/hospitals+near+me/)")

# ---------------------------
# Main Title
# ---------------------------
st.title("💚 AI Health Monitoring System")

# ---------------------------
# Model Comparison
# ---------------------------
st.subheader("📊 Model Comparison")

perf_df = pd.DataFrame({
    "Model": list(results.keys()),
    "MAE": [results[m]["MAE"] for m in results],
    "R2 Score": [results[m]["R2"] for m in results]
})

st.dataframe(perf_df)
st.success(f"Best Model: {best_model_name}")

# ---------------------------
# Input Section
# ---------------------------
st.subheader("🧍 Enter Your Health Data")

age = st.slider("Age", 18, 60, 25)
steps = st.slider("Steps", 1000, 15000, 5000)
sleep = st.slider("Sleep Hours", 4.0, 9.0, 7.0)
water = st.slider("Water Intake (L)", 1.0, 4.0, 2.5)
heart = st.slider("Heart Rate", 60, 100, 75)

# ---------------------------
# Prediction
# ---------------------------
if st.button("Predict Health Score"):

    user_df = pd.DataFrame({
        "age": [age],
        "steps": [steps],
        "sleep_hours": [sleep],
        "water_intake": [water],
        "heart_rate": [heart]
    })

    prediction = best_model.predict(user_df)[0]

    st.subheader(f"🧠 Health Score: {round(prediction, 2)} / 100")

    # Risk
    if prediction > 75:
        st.success("Low Risk 🟢")
    elif prediction > 50:
        st.warning("Moderate Risk 🟡")
    else:
        st.error("High Risk 🔴")

    # Recommendations
    st.subheader("🤖 Recommendations")
    messages = []

    if steps < 8000:
        messages.append("Walk more today!")
    if sleep < 7:
        messages.append("Sleep at least 7 hours!")
    if water < 2:
        messages.append("Drink more water!")
    if heart > 85:
        messages.append("Relax and control heart rate!")

    for msg in messages:
        st.warning(msg)
        speak(msg)

    # Feature Importance
    if hasattr(best_model, "feature_importances_"):
        st.subheader("📊 Feature Importance")
        fig, ax = plt.subplots()
        ax.barh(X.columns, best_model.feature_importances_)
        st.pyplot(fig)

    # Save History
    history = user_df.copy()
    history["score"] = prediction

    if os.path.exists("history.csv"):
        history.to_csv("history.csv", mode='a', header=False, index=False)
    else:
        history.to_csv("history.csv", index=False)

    st.success("Data Saved!")

# ---------------------------
# History Section
# ---------------------------
st.subheader("📁 History")

if os.path.exists("history.csv"):
    hist = pd.read_csv("history.csv")
    st.dataframe(hist)

    st.subheader("📈 Progress")
    st.line_chart(hist["score"])

    # Smart Reminder
    st.subheader("🔔 Smart Reminder")

    latest = hist.iloc[-1]

    reminders = []

    if latest["water_intake"] < 2:
        reminders.append("Drink more water today!")
    if latest["steps"] < 8000:
        reminders.append("Walk more today!")
    if latest["sleep_hours"] < 7:
        reminders.append("Sleep properly tonight!")

    if reminders:
        for r in reminders:
            st.warning(r)
            speak(r)
    else:
        st.success("You're doing great!")

else:
    st.info("No history yet.")

# ---------------------------
# Doctor Consultation
# ---------------------------
st.subheader("👩‍⚕️ Doctor Consultation")

if st.button("Consult Doctor"):
    st.markdown("[Click here to consult a doctor](https://www.practo.com/)")