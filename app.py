import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page setup
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# CSS styling
st.markdown("""
<style>
.main {
    background-color: #0e1117;
}

.card {
    background-color: #161b22;
    padding: 1.4rem;
    border-radius: 16px;
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1rem;
}

.price {
    font-size: 2.4rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.muted {
    opacity: 0.7;
    font-size: 0.9rem;
}
</style>
""", unsafe_allow_html=True)


# Load model
@st.cache_resource
def load_model():
    return joblib.load("best_car_price_model.pkl")


def get_expected_columns(model):
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None


def build_feature_frame_from_inputs(expected_cols, inputs):
    data = {c: 0 for c in expected_cols}
    for k, v in inputs.items():
        if k in data:
            data[k] = v
    return pd.DataFrame([data], columns=expected_cols)


def coerce_numeric_safe(s, default=np.nan):
    try:
        return float(s)
    except:
        return default


def try_predict(model, X):
    return np.array(model.predict(X)).reshape(-1)


def add_engineered_features(df):
    out = df.copy()

    if "Doors_group_le_4" not in out.columns and "Doors" in out.columns:
        doors_num = (
            out["Doors"]
            .astype(str)
            .str.replace("+", "", regex=False)
            .replace("nan", np.nan)
        )
        doors_num = pd.to_numeric(doors_num, errors="coerce")
        out["Doors_group_le_4"] = (doors_num <= 4).astype(int).fillna(0).astype(int)

    drive_col = None
    for c in ["Drive wheels", "Drive_wheels", "DriveWheels"]:
        if c in out.columns:
            drive_col = c
            break

    if drive_col is not None:
        if "Drive wheels_fwd" not in out.columns:
            out["Drive wheels_fwd"] = out[drive_col].astype(str).str.lower().isin(["front", "fwd"]).astype(int)
        if "Drive wheels_rwd" not in out.columns:
            out["Drive wheels_rwd"] = out[drive_col].astype(str).str.lower().isin(["rear", "rwd"]).astype(int)

    return out


# Header
st.markdown("## Car Price Predictor")
st.markdown(
    "<p class='muted'>Predict used car prices using a trained machine learning model</p>",
    unsafe_allow_html=True
)


# Load trained model
try:
    model = load_model()
except Exception as e:
    st.error("Model file could not be loaded. Ensure best_car_price_model.pkl is in the same folder.")
    st.code(str(e))
    st.stop()

expected_cols = get_expected_columns(model)

with st.sidebar:
    st.header("Settings")
    is_log_model = st.toggle("Model predicts log(price)", value=False)

    st.write("Model file")
    st.code("best_car_price_model.pkl")

    if expected_cols is None:
        st.error("feature_names_in_ not found in model")
    else:
        st.success("Expected feature columns detected")
        with st.expander("Show expected columns"):
            st.code(expected_cols)

if expected_cols is None:
    st.stop()


tab1, tab2, tab3 = st.tabs(["Single Predict", "Batch Predict", "Help"])


with tab1:
    st.subheader("Single Prediction")

    left, mid, right = st.columns(3)

    with left:
        prod_year = st.number_input("Production year", 1980, 2035, 2015)
        mileage = st.number_input("Mileage", 0, 999999, 80000, step=1000)
        engine_volume = st.number_input("Engine volume", 0.1, 10.0, 1.8, step=0.1)

    with mid:
        airbags = st.number_input("Airbags", 0, 30, 6)
        doors = st.selectbox("Doors", [2, 3, 4, 5])
        drive_wheels = st.selectbox("Drive wheels", ["Front", "Rear", "4x4"])

    with right:
        st.write("Category encoded value")
        category_encoded = st.text_input("Encoded category (integer)", "0")
        category_encoded_val = int(coerce_numeric_safe(category_encoded, 0))
        show_debug = st.checkbox("Show technical details", value=False)

    if st.button("Predict price"):
        inputs = {
            "Prod. year": prod_year,
            "Mileage": mileage,
            "Engine volume": engine_volume,
            "Airbags": airbags,
            "Category_encoded": category_encoded_val,
            "Doors_group_le_4": 1 if doors <= 4 else 0,
            "Drive wheels_fwd": 1 if drive_wheels.lower() == "front" else 0,
            "Drive wheels_rwd": 1 if drive_wheels.lower() == "rear" else 0,
        }

        X = build_feature_frame_from_inputs(expected_cols, inputs)

        try:
            pred = try_predict(model, X)[0]
            if is_log_model:
                pred = float(np.exp(pred))

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='price'>${pred:,.2f}</div>", unsafe_allow_html=True)
            st.markdown("<p class='muted'>Estimated car price</p>", unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            c1.write(f"Year: {prod_year}")
            c2.write(f"Mileage: {mileage:,}")
            c3.write(f"Engine: {engine_volume} L")

            st.markdown("</div>", unsafe_allow_html=True)

            if show_debug:
                st.write("Features sent to model")
                st.dataframe(X, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed")
            st.code(str(e))
            st.dataframe(X)


with tab2:
    st.subheader("Batch Prediction")

    uploaded = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.dataframe(df.head(), use_container_width=True)

        if st.button("Run batch prediction"):
            try:
                work = add_engineered_features(df)
                X = pd.DataFrame({c: work[c] if c in work.columns else 0 for c in expected_cols})

                preds = try_predict(model, X)
                if is_log_model:
                    preds = np.exp(preds)

                out = df.copy()
                out["predicted_price"] = preds

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.dataframe(out.head(50), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                csv = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download CSV", csv, "car_price_predictions.csv", "text/csv")

            except Exception as e:
                st.error("Batch prediction failed")
                st.code(str(e))


with tab3:
    st.subheader("Explanation")

    st.write("Your model expects engineered feature columns that must exactly match training.")
    st.write("This app reads the expected feature names from the trained model and rebuilds inputs safely.")
    st.write("Category encoding must be provided manually because the mapping is not stored in the model.")
