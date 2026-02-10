import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("Car Price Predictor")
st.write("Predict used car prices using a trained machine learning model")

# Load model
try:
    model = joblib.load("best_car_price_model.pkl")
except:
    st.error("Model file not found")
    st.stop()

# Feature list must match training
EXPECTED_COLUMNS = [
    "Prod. year",
    "Mileage",
    "Engine volume",
    "Airbags",
    "Doors_group_le_4",
    "Drive wheels_fwd",
    "Drive wheels_rwd",
    "Category_Age",
    "Category_Coupe",
    "Category_Goods wagon",
    "Category_Hatchback",
    "Category_Jeep",
    "Category_Limousine",
    "Category_Microbus",
    "Category_Minivan",
    "Category_Pickup",
    "Category_Sedan",
    "Category_Universal"
]

st.subheader("Enter car details")

prod_year = st.number_input("Production year", 1980, 2035, 2020)
mileage = st.number_input("Mileage", 0, 999999, 87000, step=1000)
engine_volume = st.number_input("Engine volume", 0.1, 10.0, 2.8, step=0.1)
airbags = st.number_input("Airbags", 0, 30, 1)

doors = st.selectbox("Doors", [2, 3, 4, 5])
drive_wheels = st.selectbox("Drive wheels", ["Front", "Rear"])

category = st.selectbox(
    "Car category",
    [
        "Age", "Coupe", "Goods wagon", "Hatchback", "Jeep",
        "Limousine", "Microbus", "Minivan",
        "Pickup", "Sedan", "Universal"
    ]
)

if st.button("Predict price"):
    data = {col: 0 for col in EXPECTED_COLUMNS}

    data["Prod. year"] = prod_year
    data["Mileage"] = mileage
    data["Engine volume"] = engine_volume
    data["Airbags"] = airbags
    data["Doors_group_le_4"] = 1 if doors <= 4 else 0
    data["Drive wheels_fwd"] = 1 if drive_wheels == "Front" else 0
    data["Drive wheels_rwd"] = 1 if drive_wheels == "Rear" else 0
    data[f"Category_{category}"] = 1

    X = pd.DataFrame([data])

    try:
        prediction = model.predict(X)[0]
        st.success(f"Estimated price: {prediction:,.2f}")
        st.write("Features used for prediction")
        st.dataframe(X)
    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
