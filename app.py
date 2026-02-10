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
    st.error("Your saved model does not contain feature_names_in_.")
    st.write("This app needs the exact feature column names used during training.")
    st.write("Fix option: retrain using a pandas DataFrame and save again, or paste your X_train columns list into the app.")
    st.stop()

expected_cols = list(model.feature_names_in_)

def get_dummy_options(prefix: str):
    cols = [c for c in expected_cols if c.startswith(prefix)]
    options = ["Base (all zeros)"] + [c.replace(prefix, "") for c in cols]
    return cols, options

category_cols, category_options = get_dummy_options("Category_")
drive_cols, drive_options = get_dummy_options("Drive wheels_")

st.subheader("Enter car details")

current_year = datetime.now().year

prod_year = st.number_input("Production year", min_value=1980, max_value=current_year, value=2018, step=1)
mileage = st.number_input("Mileage", min_value=0, max_value=2000000, value=87000, step=1000)
engine_volume = st.number_input("Engine volume", min_value=0.1, max_value=10.0, value=2.0, step=0.1)

airbags = None
if "Airbags" in expected_cols:
    airbags = st.number_input("Airbags", min_value=0, max_value=50, value=6, step=1)

cylinders = None
if "Cylinders" in expected_cols:
    cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, step=1)

doors = None
if "Doors" in expected_cols:
    doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1)

category_choice = None
if len(category_cols) > 0:
    category_choice = st.selectbox("Category", category_options, index=0)

drive_choice = None
if len(drive_cols) > 0:
    drive_choice = st.selectbox("Drive wheels", drive_options, index=0)

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

        if category_choice is not None and category_choice != "Base (all zeros)":
            col_name = "Category_" + category_choice
            if col_name in row:
                row[col_name] = 1

        if drive_choice is not None and drive_choice != "Base (all zeros)":
            col_name = "Drive wheels_" + drive_choice
            if col_name in row:
                row[col_name] = 1

        X = pd.DataFrame([row], columns=expected_cols)

        pred = model.predict(X)[0]
        st.success(f"Estimated price: {pred:,.2f}")

        if show_features:
            st.dataframe(X)

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
        st.write("Expected columns from model")
        st.write(expected_cols)
