import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model & scalers
reg = joblib.load("RandomForest_model.pkl")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
encoders = joblib.load("encoder.pkl")

st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Car Price Prediction App")
st.write("Enter Car Details to Predict the Price")

# Input form
with st.form("car_form"):
    vehical_name = st.text_input("Vehicle Name", "")
    Registration_Year = st.text_input("Registration Year", "")
    Insurance = st.text_input("Insurance")
    Fuel_Type = st.text_input("Fuel Type", "")
    Seats = st.text_input("Seats", "")
    Kms_Driven = st.text_input("Kms Driven", "")
    Ownership = st.text_input("Ownership", "")
    Engine_Displacement = st.text_input("Engine Displacement (cc)", "")
    Transmission = st.text_input("Transmission", "")
    Year_of_Manufacture = st.text_input("Year of Manufacture", "")
    Power = st.text_input("Power (bhp)", "")
    Drive_Type = st.text_input("Drive Type", "")
    Mileage = st.text_input("Mileage (km/l)", "")
    new_vehical_price = st.text_input("New Vehicle Price (₹ Lakhs)", "")

    submit = st.form_submit_button("Predict Price")

if submit:
    try:
        # Create DataFrame like Flask logic
        form_dict = {
            "vehical_name": vehical_name,
            "Registration Year": Registration_Year,
            "Insurance": Insurance,
            "Fuel Type": Fuel_Type,
            "Seats": Seats,
            "Kms Driven": Kms_Driven,
            "Ownership": Ownership,
            "Engine Displacement": Engine_Displacement,
            "Transmission": Transmission,
            "Year of Manufacture": Year_of_Manufacture,
            "Power": Power,
            "Drive Type": Drive_Type,
            "Mileage": Mileage,
            "new_vehical_price": new_vehical_price,
        }

        df = pd.DataFrame([form_dict])

        # Numeric columns
        numeric_cols = [
            "Registration Year","Year of Manufacture","Seats","Kms Driven",
            "Engine Displacement","Power","Mileage","new_vehical_price"
        ]

        for col in numeric_cols:
            df[col] = df[col].astype(str).str.replace(",", "").str.extract(r"(\d+\.?\d*)")
            df[col] = df[col].astype(float).fillna(0)

        # Categorical columns
        cat_cols = ["Ownership","Fuel Type","Transmission","Drive Type","vehical_name","Insurance"]

        for col in cat_cols:
            le = encoders[col]
            df[col] = df[col].apply(lambda x: x if x in le.classes_ else "unknown")
            if "unknown" not in le.classes_:
                le.classes_ = np.append(le.classes_, "unknown")
            df[col] = le.transform(df[col])

        # Scale numeric
        df[numeric_cols] = scaler_x.transform(df[numeric_cols])

        # Predict
        feature_cols = numeric_cols + cat_cols
        X_new = df[feature_cols]
        y_pred_scaled = reg.predict(X_new)
        y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

        st.success(f"💰 Predicted Vehicle Price: ₹{y_pred[0][0]:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")

