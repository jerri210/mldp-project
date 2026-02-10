import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("Car Price Predictor")
st.write("Predict used car prices using a trained machine learning model")

# Load trained model
try:
    model = joblib.load("best_car_price_model.pkl")
except Exception as e:
    st.error("Model file not found. Ensure best_car_price_model.pkl is in the same folder.")
    st.stop()

# These MUST match training exactly
EXPECTED_COLUMNS = [
    "Car_Age",
    "Category_encoded",
    "Cylinders",
    "Doors",
    "Engine_per_Age"
]

st.subheader("Enter car details")

current_year = datetime.now().year

prod_year = st.number_input("Production year", min_value=1980, max_value=current_year, value=2018)
engine_volume = st.number_input("Engine volume", min_value=0.5, max_value=10.0, value=2.0, step=0.1)
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, step=1)
doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1)
category_encoded = st.number_input("Category encoded value", min_value=0, value=0, step=1)

if st.button("Predict price"):
    try:
        car_age = current_year - prod_year
        if car_age <= 0:
            car_age = 1

        engine_per_age = engine_volume / car_age

        data = {
            "Car_Age": car_age,
            "Category_encoded": category_encoded,
            "Cylinders": cylinders,
            "Doors": doors,
            "Engine_per_Age": engine_per_age
        }

        X = pd.DataFrame([[data[col] for col in EXPECTED_COLUMNS]], columns=EXPECTED_COLUMNS)

        prediction = model.predict(X)[0]

        st.success(f"Estimated price: {prediction:,.2f}")
        st.write("Features sent to model")
        st.dataframe(X)

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
