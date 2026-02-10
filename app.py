import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Car Price Predictor", layout="wide")

def reset_inputs():
    keys_to_clear = [
        "prod_year", "mileage", "engine_volume", "airbags", "cylinders", "doors",
        "category_choice", "drive_choice", "fuel_choice", "gear_choice",
        "leather_yes", "rhd_yes", "levy", "show_debug", "show_engineered"
    ]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
    st.rerun()

st.title("Car Price Predictor")
st.caption("Predict used car prices using a trained machine learning model")

try:
    model = joblib.load("best_car_price_model.pkl")
except Exception as e:
    st.error("Could not load best_car_price_model.pkl. Put it in the same folder as app.py.")
    st.code(str(e))
    st.stop()

if not hasattr(model, "feature_names_in_"):
    st.error("Your model does not contain feature_names_in_.")
    st.write("This app needs the model's expected feature names to build the correct input columns.")
    st.stop()

expected_cols = list(model.feature_names_in_)

def dummy_cols(prefix: str):
    return [c for c in expected_cols if c.startswith(prefix)]

def dropdown_from_prefix(label: str, prefix: str, state_key: str):
    cols = dummy_cols(prefix)
    if not cols:
        return None, []
    options = ["Not selected"] + [c.replace(prefix, "") for c in cols]
    choice = st.selectbox(label, options, index=0, key=state_key)
    return choice, cols

with st.sidebar:
    st.header("Prediction options")
    show_debug = st.toggle("Show full feature vector (debug)", value=False, key="show_debug")
    show_engineered = st.toggle("Show engineered fields", value=True, key="show_engineered")
    st.button("Reset input", type="secondary", on_click=reset_inputs)

current_year = datetime.now().year

st.subheader("Enter car details")

colA, colB, colC = st.columns(3)

with colA:
    prod_year = st.number_input(
        "Production year", min_value=1980, max_value=current_year, value=2018, step=1, key="prod_year"
    )
    mileage = st.number_input(
        "Mileage", min_value=0, max_value=2000000, value=87000, step=1000, key="mileage"
    )
    engine_volume = st.number_input(
        "Engine volume", min_value=0.1, max_value=10.0, value=2.0, step=0.1, key="engine_volume"
    )

with colB:
    airbags = None
    if "Airbags" in expected_cols:
        airbags = st.number_input("Airbags", min_value=0, max_value=50, value=6, step=1, key="airbags")

    cylinders = None
    if "Cylinders" in expected_cols:
        cylinders = st.number_input("Cylinders", min_value=2, max_value=16, value=4, step=1, key="cylinders")

    doors = None
    if "Doors" in expected_cols:
        doors = st.number_input("Doors", min_value=2, max_value=6, value=4, step=1, key="doors")

with colC:
    category_choice, _ = dropdown_from_prefix("Category", "Category_", "category_choice")
    drive_choice, _ = dropdown_from_prefix("Drive wheels", "Drive wheels_", "drive_choice")
    fuel_choice, _ = dropdown_from_prefix("Fuel type", "Fuel type_", "fuel_choice")
    gear_choice, _ = dropdown_from_prefix("Gear box type", "Gear box type_", "gear_choice")

with st.expander("Additional options", expanded=False):
    left, right = st.columns(2)

    with left:
        leather_yes = False
        if "Leather interior_Yes" in expected_cols:
            leather_yes = st.toggle("Leather interior = Yes", value=False, key="leather_yes")

        rhd_yes = False
        if "Wheel_Right-hand drive" in expected_cols:
            rhd_yes = st.toggle("Right-hand drive = Yes", value=False, key="rhd_yes")

    with right:
        levy = None
        if "Levy" in expected_cols:
            levy = st.number_input("Levy", min_value=0, max_value=1000000, value=0, step=50, key="levy")

predict_clicked = st.button("Predict", type="primary")

def build_full_feature_row():
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

    if "Doors_group_le_4" in row and doors is not None:
        row["Doors_group_le_4"] = 1 if int(doors) <= 4 else 0

    if "Leather interior_Yes" in row:
        row["Leather interior_Yes"] = 1 if leather_yes else 0

    if "Wheel_Right-hand drive" in row:
        row["Wheel_Right-hand drive"] = 1 if rhd_yes else 0

    if "Levy" in row and levy is not None:
        row["Levy"] = float(levy)

    def set_one_hot(prefix, choice):
        if choice and choice != "Not selected":
            col = prefix + choice
            if col in row:
                row[col] = 1

    set_one_hot("Category_", category_choice)
    set_one_hot("Drive wheels_", drive_choice)
    set_one_hot("Fuel type_", fuel_choice)
    set_one_hot("Gear box type_", gear_choice)

    return row

def build_selected_display(row_dict):
    selected = {}

    selected["Production year"] = int(prod_year)
    selected["Mileage"] = int(mileage)
    selected["Engine volume"] = float(engine_volume)

    if airbags is not None:
        selected["Airbags"] = int(airbags)
    if cylinders is not None:
        selected["Cylinders"] = int(cylinders)
    if doors is not None:
        selected["Doors"] = int(doors)

    if category_choice and category_choice != "Not selected":
        selected["Category"] = category_choice
    if drive_choice and drive_choice != "Not selected":
        selected["Drive wheels"] = drive_choice
    if fuel_choice and fuel_choice != "Not selected":
        selected["Fuel type"] = fuel_choice
    if gear_choice and gear_choice != "Not selected":
        selected["Gear box type"] = gear_choice

    if "Leather interior_Yes" in expected_cols and leather_yes:
        selected["Leather interior"] = "Yes"
    if "Wheel_Right-hand drive" in expected_cols and rhd_yes:
        selected["Right-hand drive"] = "Yes"
    if "Levy" in expected_cols and levy is not None and levy != 0:
        selected["Levy"] = float(levy)

    if show_engineered:
        if "Car_Age" in row_dict:
            selected["Car_Age"] = row_dict["Car_Age"]
        if "Mileage_log" in row_dict:
            selected["Mileage_log"] = round(float(row_dict["Mileage_log"]), 4)
        if "Engine_per_Age" in row_dict:
            selected["Engine_per_Age"] = round(float(row_dict["Engine_per_Age"]), 4)
        if "Doors_group_le_4" in row_dict:
            selected["Doors_group_le_4"] = int(row_dict["Doors_group_le_4"])

    df = pd.DataFrame([selected]).T.reset_index()
    df.columns = ["Field", "Value"]
    return df

if predict_clicked:
    try:
        row = build_full_feature_row()
        X = pd.DataFrame([row], columns=expected_cols)

        pred = model.predict(X)[0]

        st.subheader("Result")
        st.markdown(f"### Estimated price: **${pred:,.2f}**")

        st.subheader("Selected inputs")
        st.dataframe(build_selected_display(row), use_container_width=True)

        if show_debug:
            st.subheader("Full feature vector (debug)")
            st.dataframe(X, use_container_width=True)

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
