import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("Car Price Predictor")
st.write("Predict used car prices using a trained machine learning model")

# Load model
try:
    model = joblib.load("best_car_price_model.pkl")
except Exception as e:
    st.error("Model file not found. Make sure best_car_price_model.pkl is in the same folder.")
    st.stop()

# Hardcoded feature list
EXPECTED_COLUMNS = [
    "Prod. year",
    "Mileage",
    "Engine volume",
    "Airbags",
    "Category_encoded",
    "Doors_group_le_4",
    "Drive wheels_fwd",
    "Drive wheels_rwd"
]

st.subheader("Enter car details")

prod_year = st.number_input("Production year", 1980, 2035, 2015)
mileage = st.number_input("Mileage", 0, 999999, 80000, step=1000)
engine_volume = st.number_input("Engine volume", 0.1, 10.0, 1.8, step=0.1)
airbags = st.number_input("Airbags", 0, 30, 6)

doors = st.selectbox("Doors", [2, 3, 4, 5])
drive_wheels = st.selectbox("Drive wheels", ["Front", "Rear", "4x4"])

category_encoded = st.number_input("Category encoded value", value=0, step=1)

if st.button("Predict price"):
    data = {
        "Prod. year": prod_year,
        "Mileage": mileage,
        "Engine volume": engine_volume,
        "Airbags": airbags,
        "Category_encoded": category_encoded,
        "Doors_group_le_4": 1 if doors <= 4 else 0,
        "Drive wheels_fwd": 1 if drive_wheels == "Front" else 0,
        "Drive wheels_rwd": 1 if drive_wheels == "Rear" else 0,
    }

    X = pd.DataFrame([[data[col] for col in EXPECTED_COLUMNS]], columns=EXPECTED_COLUMNS)

    try:
        prediction = model.predict(X)[0]
        st.success(f"Estimated price: {prediction:,.2f}")
        st.write("Input features used for prediction")
        st.dataframe(X)
    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
