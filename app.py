import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Page setup
# ----------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="wide")

# ----------------------------
# Load model (ONLY 1 file)
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("best_car_price_model.pkl")

def get_expected_columns(model):
    """
    Many sklearn models store feature names used during fit.
    If your model has feature_names_in_, we can auto-match perfectly.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    return None

def build_feature_frame_from_inputs(expected_cols, inputs: dict):
    """
    Build a 1-row DataFrame that matches expected_cols exactly.
    Any missing expected columns will be filled with 0.
    """
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
    preds = model.predict(X)
    return np.array(preds).reshape(-1)

def add_engineered_features(df: pd.DataFrame):
    """
    Create engineered columns if possible, based on common raw columns.
    We do NOT guess Category_encoded mapping (user must provide encoded ints if needed).
    """
    out = df.copy()

    # Doors_group_le_4
    if "Doors_group_le_4" not in out.columns and "Doors" in out.columns:
        # Convert Doors like "4" / "5+" to numeric where possible
        doors_num = (
            out["Doors"]
            .astype(str)
            .str.replace("+", "", regex=False)
            .replace("nan", np.nan)
        )
        doors_num = pd.to_numeric(doors_num, errors="coerce")
        out["Doors_group_le_4"] = (doors_num <= 4).astype(int).fillna(0).astype(int)

    # Drive wheels_fwd / Drive wheels_rwd
    # Try raw column "Drive wheels" or "Drive wheels" variants
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

# ----------------------------
# UI Header
# ----------------------------
st.title("🚗 Car Price Predictor")
st.caption("Uses ONLY your saved model file: best_car_price_model.pkl")

# ----------------------------
# Load model
# ----------------------------
try:
    model = load_model()
except Exception as e:
    st.error("Could not load best_car_price_model.pkl. Put it in the same folder as app.py.")
    st.code(str(e))
    st.stop()

expected_cols = get_expected_columns(model)

with st.sidebar:
    st.header("⚙️ Settings")
    is_log_model = st.toggle("Model predicts log(price)", value=False)
    st.markdown("---")
    st.write("Model file:")
    st.code("best_car_price_model.pkl")

    if expected_cols is None:
        st.error("Your model does NOT expose feature_names_in_.")
        st.write("This app needs feature_names_in_ to perfectly match columns with only 1 .pkl file.")
    else:
        st.success("Expected columns detected ✅")
        st.caption("Your model expects these columns (used during fit):")
        with st.expander("Show expected columns"):
            st.code(expected_cols)

if expected_cols is None:
    st.stop()

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["🔮 Single Predict", "📁 Batch Predict (CSV)", "ℹ️ Help"])

# ==========================================================
# TAB 1: Single prediction
# ==========================================================
with tab1:
    st.subheader("Single Prediction")

    left, mid, right = st.columns([1, 1, 1])

    # We only show fields that are likely needed based on your error
    # AND we only apply them if the column exists in expected_cols.
    with left:
        prod_year = st.number_input("Prod. year", min_value=1980, max_value=2035, value=2015, step=1)
        mileage = st.number_input("Mileage", min_value=0, max_value=999999, value=80000, step=1000)
        engine_volume = st.number_input("Engine volume", min_value=0.1, max_value=10.0, value=1.8, step=0.1)

    with mid:
        airbags = st.number_input("Airbags", min_value=0, max_value=30, value=6, step=1)
        doors = st.selectbox("Doors", [2, 3, 4, 5], index=2)
        drive_wheels = st.selectbox("Drive wheels", ["Front", "Rear", "4x4"], index=0)

    with right:
        st.write("Category_encoded")
        st.caption("Your model expects `Category_encoded`, but we don't know your encoding map.")
        category_encoded = st.text_input("Enter encoded category (integer)", value="0")
        category_encoded_val = int(coerce_numeric_safe(category_encoded, default=0))

        st.markdown("---")
        show_debug = st.checkbox("Show debug (features sent to model)", value=True)

    if st.button("Predict Price", type="primary"):
        # Build engineered values
        doors_group_le_4 = 1 if doors <= 4 else 0
        drive_fwd = 1 if drive_wheels.lower() == "front" else 0
        drive_rwd = 1 if drive_wheels.lower() == "rear" else 0

        # Inputs dict: we only set them if they exist in expected columns
        inputs = {
            "Prod. year": prod_year,
            "Mileage": mileage,
            "Engine volume": engine_volume,
            "Airbags": airbags,
            "Category_encoded": category_encoded_val,
            "Doors_group_le_4": doors_group_le_4,
            "Drive wheels_fwd": drive_fwd,
            "Drive wheels_rwd": drive_rwd,
        }

        X = build_feature_frame_from_inputs(expected_cols, inputs)

        try:
            pred = try_predict(model, X)[0]
            if is_log_model:
                pred = float(np.exp(pred))

            st.success("Prediction successful ✅")

            a, b, c = st.columns(3)
            a.metric("Estimated Price", f"{pred:,.2f}")
            b.metric("Prod. year", f"{int(prod_year)}")
            c.metric("Mileage", f"{int(mileage):,}")

            if show_debug:
                st.markdown("### Features sent to model")
                st.dataframe(X, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed. This usually means some expected features are still not being provided correctly.")
            st.code(str(e))
            st.markdown("### Features we sent")
            st.dataframe(X, use_container_width=True)

# ==========================================================
# TAB 2: Batch prediction
# ==========================================================
with tab2:
    st.subheader("Batch Prediction (CSV)")

    st.write("Upload a CSV and we’ll predict for each row. Best case: your CSV already has the engineered columns.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.markdown("### Preview")
        st.dataframe(df.head(30), use_container_width=True)

        if st.button("Run Batch Prediction", type="primary"):
            try:
                work = add_engineered_features(df)

                # Build X that matches expected_cols exactly:
                # - If work has a column, use it
                # - Else fill 0
                X = pd.DataFrame({c: work[c] if c in work.columns else 0 for c in expected_cols})

                # Predict
                preds = try_predict(model, X)
                if is_log_model:
                    preds = np.exp(preds)

                out = df.copy()
                out["predicted_price"] = preds

                st.success("Batch prediction successful ✅")
                st.dataframe(out.head(50), use_container_width=True)

                st.markdown("### Predicted price distribution")
                st.bar_chart(out["predicted_price"])

                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download predictions CSV",
                    data=csv_bytes,
                    file_name="car_price_predictions.csv",
                    mime="text/csv",
                )

            except Exception as e:
                st.error("Batch prediction failed.")
                st.code(str(e))

                st.markdown("### Debug info")
                st.write("These are the columns your model expects:")
                st.code(expected_cols)

                st.write("These are the columns in your uploaded CSV:")
                st.code(list(df.columns))

                st.info(
                    "If your model expects columns like `Category_encoded`, "
                    "your CSV must include them (as encoded integers), or you must compute them exactly the same way as training."
                )

# ==========================================================
# TAB 3: Help
# ==========================================================
with tab3:
    st.subheader("Why your earlier version failed")

    st.write(
        """
Your error showed your model expects engineered columns like:
- Airbags
- Category_encoded
- Doors_group_le_4
- Drive wheels_fwd
- Drive wheels_rwd

But your app was sending raw columns like:
- Color, Doors, Fuel type, Gear box type

So sklearn rejected the input because **feature names must match training**.

### What this app does instead
It reads the **exact expected feature names** from your model and always builds an input table that matches them.

### About Category_encoded
Because you only have 1 file (the model), we don't know the exact encoding map you used during training.
So:
- for Single Predict: you type the integer
- for Batch Predict: your CSV should already include Category_encoded, OR you need to recreate the same encoding rule.
        """
    )
