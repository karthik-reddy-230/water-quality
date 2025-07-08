# app.py

# ===============================
# 💧 Water Pollutants Predictor
# ===============================

import pandas as pd
import numpy as np
import joblib
import streamlit as st

# Load model and columns
model = joblib.load("pollution_model.pkl")
model_cols = joblib.load("model_columns.pkl")

# Streamlit UI
st.title("💧 Water Pollutants Predictor")
st.write("Predict key water pollutants based on Year and Station ID")

# User input
year_input = st.number_input("Enter Year", min_value=2000, max_value=2100, value=2022)
station_id = st.text_input("Enter Station ID", value='1')

# Prediction
if st.button('Predict'):
    if not station_id.strip():
        st.warning('🚨 Please enter a valid Station ID.')
    else:
        # Prepare data
        input_df = pd.DataFrame({'year': [year_input], 'id': [station_id]})
        input_encoded = pd.get_dummies(input_df, columns=['id'])

        # Align with model columns
        for col in model_cols:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[model_cols]

        # Predict
        try:
            predicted_pollutants = model.predict(input_encoded)[0]
            pollutants = ['O2', 'NO3', 'NO2', 'SO4', 'PO4', 'CL']

            st.subheader(f"🔬 Predicted pollutant levels for station '{station_id}' in {year_input}:")
            for p, val in zip(pollutants, predicted_pollutants):
                st.write(f"**{p}**: {val:.2f}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

# ===============================
# README Info (optional)
# ===============================

README = """
# 💧 Water Pollutants Predictor

This Streamlit app predicts key water pollutant levels based on the **Year** and **Station ID**. It uses a trained machine learning model to estimate the concentrations of pollutants like O₂, NO₃, NO₂, SO₄, PO₄, and Cl.

## 🚀 Features

- Input **Year** and **Station ID** to get predictions.
- Predicts levels of:
  - Oxygen (O₂)
  - Nitrate (NO₃)
  - Nitrite (NO₂)
  - Sulfate (SO₄)
  - Phosphate (PO₄)
  - Chloride (Cl)
- User-friendly interface built with Streamlit.

## 📁 Project Structure

