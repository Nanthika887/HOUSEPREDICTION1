import streamlit as st
import pickle
import numpy as np

# Load Model
with open("model_pickle", "rb") as f:
    model = pickle.load(f)

# Page Config
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="centered"
)

# Title
st.title("🏠 House Price Prediction App")
st.write("Enter house details and predict the price")

# Inputs
area = st.number_input("Enter Area (sq ft)", min_value=100, value=1000)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, value=2)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, value=2)

# Predict Button
if st.button("Predict Price"):

    input_data = np.array([[area, bedrooms, bathrooms]])

    prediction = model.predict(input_data)

    st.success(
        f"Estimated House Price: ₹ {prediction[0]:,.2f}"
    )

# Footer
st.markdown("---")
st.caption("Created using Streamlit + Elastic Net")
