import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os

# Load model safely
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "model.pkl")

model = pickle.load(open(model_path, "rb"))

st.set_page_config(page_title="Home Price Predictor")

st.title("🏠 Home Price Prediction App")
st.write("Enter house details to predict price")

# Inputs
area = st.number_input("Area (sq ft)", min_value=200, max_value=10000, step=50)
bhk = st.number_input("BHK", min_value=1, max_value=10)
bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10)
parking = st.number_input("Parking", min_value=1, max_value=3)

# Predict
if st.button("Predict Price"):
    features = np.array([[area, bhk, bathrooms, parking]])
    prediction = model.predict(features)

    st.success(f"💰 Estimated Price: ₹ {round(prediction[0],2)}")
