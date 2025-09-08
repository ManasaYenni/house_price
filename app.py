# app.py

import streamlit as st
import pickle
import numpy as np

# Load trained model and r² score
with open("model/house_model.pkl", "rb") as f:
    model, r2_score = pickle.load(f)

st.set_page_config(page_title="House Price Predictor")
st.title("🏠 House Price Predictor")

st.markdown("Enter property details below to estimate the house price:")

# Input widgets
area = st.number_input("Area (sq ft)", min_value=200, step=10)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, step=1)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, step=1)

# Predict button
if st.button("Predict Price"):
    features = np.array([[area, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    st.success(f"Estimated House Price: ₹ {prediction:,.2f}")

# Show model accuracy
st.info(f"🔍 Model Accuracy (R² Score): {r2_score:.2f}")
