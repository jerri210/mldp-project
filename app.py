import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

st.set_page_config(page_title="Car Price Predictor", layout="wide")

# Reset inputs
def reset_inputs():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

# Light styling
st.markdown(
    """
<style>
/* Page width + spacing */
.block-container { padding-top: 1.2rem; }

/* Subtle card */
.card{
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 16px 18px;
  background: rgba(255,255,255,0.03);
  margin-bottom: 14px;
}

/* Result highlight */
.result-card{
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  padding: 18px 20px;
  background: linear-gradient(135deg, rgba(0, 180, 255, 0.12), rgba(0, 255, 180, 0.10));
  margin-top: 0.5rem;
}
.result-title{ font-size: 0.95rem; opacity: 0.85; font-weight: 700; }
.result-value{ font-size: 2rem; font-weight: 900; margin-top: 2px; }

/* Small helper text */
.small-note { font-size: 0.92rem; opacity: 0.85; }

/* Sidebar spacing */
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
</style>
""",
    unsafe_allow_html=True,
)

# Header + banner
try:
    st.image("car banner.jpeg", use_container_width=True)
except Exception:
    # If image missing, don't crash the app
    st.info("Tip: Put **car banner.jpeg** in the same folder as app.py to show the banner image.")

st.title("🚗 Car Price Predictor")
st.caption("Estimate used car prices instantly using your trained ML model.")

# Load model
try:
    model = joblib.load("best_car_price_model.pkl")
except Exception:
    st.error("Could not load best_car_price_model.pkl. Make sure it is in the same folder.")
    st.stop()

# Ensure model has feature names
if not hasattr(model, "feature_names_in_"):
    st.error("Model does not expose feature_names_in_.")
    st.stop()

expected_cols = list(model.feature_names_in_)

# Friendly labels for short codes
LABEL_MAP = {
    "fwd": "Front-wheel drive",
    "rwd": "Rear-wheel drive",
    "Manual": "Manual transmission",
    "Tiptronic": "Automatic (Tiptronic)",
    "Variator": "Automatic (CVT)",
    "Petrol": "Petrol",
    "Diesel": "Diesel",
    "Hybrid": "Hybrid",
    "Plug-in Hybrid": "Plug-in Hybrid",
    "Hydrogen": "Hydrogen",
    "LPG": "LPG",
}

def dummy_cols(prefix):
    return [c for c in expected_cols if c.startswith(prefix)]

def dropdown_from_prefix(label, prefix, key, help_text):
    cols = dummy_cols(prefix)
    if not cols:
        return None

    raw_values = [c.replace(prefix, "") for c in cols]

    display_options = ["Not selected"]
    value_map = {"Not selected": None}

    for v in raw_values:
        display = LABEL_MAP.get(v, v)
        display_options.append(display)
        value_map[display] = v

    chosen_display = st.selectbox(
        label,
        display_options,
        index=0,
        key=key,
        help=help_text,
    )

    return value_map[chosen_display]

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    show_engineered = st.toggle(
        "Show calculated fields",
        value=True,
        help="Shows values calculated from your inputs, such as car age.",
    )
    show_debug = st.toggle(
        "Show technical details",
        value=False,
        help="Shows the full feature vector sent to the model.",
    )
    st.divider()
    st.button("🔄 Reset inputs", on_click=reset_inputs, use_container_width=True)

# Main inputs
current_year = datetime.now().year

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Enter car details")
st.markdown('<div class="small-note">Fill in what you know — unknown fields can be left at default.</div>',
            unsafe_allow_html=True)

c1, c2, c3 = st.columns(3)

with c1:
    prod_year = st.number_input(
        "Production year",
        1980,
        current_year,
        2018,
        help="The year the car was manufactured.",
    )
    mileage = st.number_input(
        "Mileage (km)",
        0,
        2_000_000,
        87_000,
        step=1000,
        help="Total distance the car has been driven (km).",
    )
    engine_volume = st.number_input(
        "Engine volume (L)",
        0.1,
        10.0,
        2.0,
        step=0.1,
        help="Engine size in litres (for example 1.6L or 2.0L).",
    )

with c2:
    airbags = (
        st.number_input(
            "Airbags",
            0,
            50,
            6,
            help="Number of airbags installed in the car.",
        )
        if "Airbags" in expected_cols
        else None
    )

    cylinders = (
        st.number_input(
            "Cylinders",
            2,
            16,
            4,
            help="Number of cylinders in the engine.",
        )
        if "Cylinders" in expected_cols
        else None
    )

    doors = (
        st.number_input(
            "Doors",
            2,
            6,
            4,
            help="Number of doors on the car.",
        )
        if "Doors" in expected_cols
        else None
    )

with c3:
    category_choice = dropdown_from_prefix(
        "Category",
        "Category_",
        "category",
        "Body type of the car such as Sedan, Jeep or Hatchback.",
    )
    drive_choice = dropdown_from_prefix(
        "Drive wheels",
        "Drive wheels_",
        "drive",
        "Which wheels receive power from the engine.",
    )
    fuel_choice = dropdown_from_prefix(
        "Fuel type",
        "Fuel type_",
        "fuel",
        "Main fuel used by the car.",
    )
    gear_choice = dropdown_from_prefix(
        "Gear box type",
        "Gear box type_",
        "gear",
        "Type of transmission used by the car.",
    )

st.markdown("</div>", unsafe_allow_html=True)

# Additional options
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Additional options")

leather_yes = (
    st.toggle(
        "Leather interior",
        value=False,
        help="Turn on if the car has leather seats.",
    )
    if "Leather interior_Yes" in expected_cols
    else False
)

rhd_yes = (
    st.toggle(
        "Right-hand drive",
        value=False,
        help="Turn on if the steering wheel is on the right side.",
    )
    if "Wheel_Right-hand drive" in expected_cols
    else False
)

levy = (
    st.number_input(
        "Levy",
        0,
        1_000_000,
        0,
        step=50,
        help="Extra levy value used in the dataset. Leave 0 if unsure.",
    )
    if "Levy" in expected_cols
    else None
)

st.markdown("</div>", unsafe_allow_html=True)

# Predict button
predict_col1, predict_col2 = st.columns([1, 3])
with predict_col1:
    predict_clicked = st.button("✨ Predict", type="primary", use_container_width=True)

if predict_clicked:
    try:
        row = {c: 0 for c in expected_cols}

        car_age = max(0, current_year - prod_year)

        if "Prod. year" in row:
            row["Prod. year"] = prod_year
        if "Car_Age" in row:
            row["Car_Age"] = car_age
        if "Mileage" in row:
            row["Mileage"] = mileage
        if "Mileage_log" in row:
            row["Mileage_log"] = float(np.log1p(mileage))
        if "Engine volume" in row:
            row["Engine volume"] = engine_volume
        if "Engine_per_Age" in row:
            row["Engine_per_Age"] = engine_volume / (car_age + 1)

        if airbags is not None and "Airbags" in row:
            row["Airbags"] = airbags
        if cylinders is not None and "Cylinders" in row:
            row["Cylinders"] = cylinders
        if doors is not None and "Doors" in row:
            row["Doors"] = doors
        if "Doors_group_le_4" in row and doors is not None:
            row["Doors_group_le_4"] = 1 if doors <= 4 else 0

        if "Leather interior_Yes" in row:
            row["Leather interior_Yes"] = 1 if leather_yes else 0
        if "Wheel_Right-hand drive" in row:
            row["Wheel_Right-hand drive"] = 1 if rhd_yes else 0
        if levy is not None and "Levy" in row:
            row["Levy"] = levy

        def set_one_hot(prefix, choice):
            if choice:
                col = prefix + choice
                if col in row:
                    row[col] = 1

        set_one_hot("Category_", category_choice)
        set_one_hot("Drive wheels_", drive_choice)
        set_one_hot("Fuel type_", fuel_choice)
        set_one_hot("Gear box type_", gear_choice)

        X = pd.DataFrame([row], columns=expected_cols)
        prediction = model.predict(X)[0]

        # Result section (highlight)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Result")

        st.markdown(
            f"""
            <div class="result-card">
              <div class="result-title">Estimated price</div>
              <div class="result-value">${prediction:,.2f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # quick metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Car age", f"{car_age} yrs")
        m2.metric("Mileage", f"{mileage:,} km")
        m3.metric("Engine", f"{engine_volume} L")
        m4.metric("Fuel", LABEL_MAP.get(fuel_choice, str(fuel_choice or "-")))

        st.markdown("</div>", unsafe_allow_html=True)

        # Selected inputs + debug (tabs)
        selected = {
            "Production year": prod_year,
            "Mileage (km)": mileage,
            "Engine volume (L)": engine_volume,
        }

        if airbags is not None:
            selected["Airbags"] = airbags
        if cylinders is not None:
            selected["Cylinders"] = cylinders
        if doors is not None:
            selected["Doors"] = doors
        if category_choice:
            selected["Category"] = category_choice
        if drive_choice:
            selected["Drive wheels"] = LABEL_MAP.get(drive_choice, drive_choice)
        if fuel_choice:
            selected["Fuel type"] = LABEL_MAP.get(fuel_choice, fuel_choice)
        if gear_choice:
            selected["Gear box type"] = LABEL_MAP.get(gear_choice, gear_choice)
        if leather_yes:
            selected["Leather interior"] = "Yes"
        if rhd_yes:
            selected["Right-hand drive"] = "Yes"
        if levy and levy != 0:
            selected["Levy"] = levy

        if show_engineered:
            if "Car_Age" in row:
                selected["Car age"] = row["Car_Age"]
            if "Mileage_log" in row:
                selected["Mileage (log)"] = round(row["Mileage_log"], 4)
            if "Engine_per_Age" in row:
                selected["Engine per age"] = round(row["Engine_per_Age"], 4)

        df_selected = pd.DataFrame(selected.items(), columns=["Field", "Value"])

        tab1, tab2 = st.tabs(["📋 Selected inputs", "🧪 Technical details"])
        with tab1:
            st.dataframe(df_selected, use_container_width=True)

        with tab2:
            if show_debug:
                st.dataframe(X, use_container_width=True)
            else:
                st.info("Turn on **Show technical details** in the sidebar to view the full feature vector.")

    except Exception as e:
        st.error("Prediction failed")
        st.code(str(e))
