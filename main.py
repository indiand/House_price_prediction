import streamlit as st
import sklearn
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("model.pkl", "rb"))

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
    features = np.array([[area , bhk , bathrooms , parking ]])
    prediction = model.predict(features)

    st.success(f"💰 Estimated Price: ₹ {round(prediction[0],2)}") 

 