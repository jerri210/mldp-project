import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("Car Price Predictor")
st.write("Predict used car prices using a trained machine learning model")

try:
    model = joblib.load("best_car_price_model.pkl")
except Exception as e:
    st.error("Could not load best_car_price_model.pkl. Put it in the same folder as app.py.")
    st.code(str(e))
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("Model does not have feature_names_in_.")
    st.write("This app needs the exact feature names used during training.")
    st.stop()

expected_cols = list(model.feature_names_in_)

def get_dummy_cols(prefix):
    return [c for c in expected_cols if c.startswith(prefix)]

def make_dropdown(label, prefix):
    cols = get_dummy_cols(prefix)
    if not cols:
        return None, []
    options = ["Base (all zeros)"] + [c.replace(prefix, "") for c in cols]
    choice = st.selectbox(label, options, index=0)
    return choice, cols

st.subheader("Enter car details")

current_year = datetime.now().year

prod_year = st.number_input("Production year", min_value=1980, max_value=current_year, value=2018, step=1)
mileage = st.number_input("Mileage", min_value=0, max_value=2000000, value=87000, step=1000)
engine_volume = st.number_input("Engine volume", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

airbags = st.number_input("Airbags", min_value=0, max_value=50, value=6, step=1) if "Airbags" in expected_cols else None
cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, step=1) if "Cylinders" in expected_cols else None
doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1) if "Doors" in expected_cols else None

category_choice, category_cols = make_dropdown("Category", "Category_")
drive_choice, drive_cols = make_dropdown("Drive wheels", "Drive wheels_")
fuel_choice, fuel_cols = make_dropdown("Fuel type", "Fuel type_")
gear_choice, gear_cols = make_dropdown("Gear box type", "Gear box type_")

leather_yes = st.toggle("Leather interior = Yes", value=False) if "Leather interior_Yes" in expected_cols else False
rhd_yes = st.toggle("Right-hand drive = Yes", value=False) if "Wheel_Right-hand drive" in expected_cols else False

show_features = st.checkbox("Show features sent to model", value=True)

if st.button("Predict price"):
    try:
        row = {c: 0 for c in expected_cols}

        car_age = current_year - int(prod_year)
        if car_age < 0:
            car_age = 0

        if "Car_Age" in row:
            row["Car_Age"] = car_age

        if "Prod. year" in row:
            row["Prod. year"] = int(prod_year)

        if "Mileage" in row:
            row["Mileage"] = float(mileage)

        if "Mileage_log" in row:
            row["Mileage_log"] = float(np.log1p(mileage))

        if "Engine volume" in row:
            row["Engine volume"] = float(engine_volume)

        if "Engine_per_Age" in row:
            row["Engine_per_Age"] = float(engine_volume) / (car_age + 1)

        if airbags is not None and "Airbags" in row:
            row["Airbags"] = int(airbags)

        if cylinders is not None and "Cylinders" in row:
            row["Cylinders"] = int(cylinders)

        if doors is not None and "Doors" in row:
            row["Doors"] = int(doors)

        if "Doors_group_le_4" in row:
            row["Doors_group_le_4"] = 1 if (doors is not None and int(doors) <= 4) else 0

        if "Leather interior_Yes" in row:
            row["Leather interior_Yes"] = 1 if leather_yes else 0

        if "Wheel_Right-hand drive" in row:
            row["Wheel_Right-hand drive"] = 1 if rhd_yes else 0

        if category_choice and category_choice != "Base (all zeros)":
            col = "Category_" + category_choice
            if col in row:
                row[col] = 1

        if drive_choice and drive_choice != "Base (all zeros)":
            col = "Drive wheels_" + drive_choice
            if col in row:
                row[col] = 1

        if fuel_choice and fuel_choice != "Base (all zeros)":
            col = "Fuel type_" + fuel_choice
            if col in row:
                row[col] = 1

        if gear_choice and gear_choice != "Base (all zeros)":
            col = "Gear box type_" + gear_choice
            if col in row:
                row[col] = 1

        X = pd.DataFrame([row], columns=expected_cols)

        pred = model.predict(X)[0]
        st.success(f"Estimated price: {pred:,.2f}")

        if show_features:
            st.dataframe(X)

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
