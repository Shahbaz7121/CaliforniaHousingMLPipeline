import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load("best_model.pkl")

st.title("California Housing Price Predictor")

# Collect input from user
median_income = st.number_input("Median Income", min_value=0.0, value=3.0)
total_rooms = st.number_input("Total Rooms", min_value=0.0, value=2000.0)
total_bedrooms = st.number_input("Total Bedrooms", min_value=0.0, value=400.0)
population = st.number_input("Population", min_value=0.0, value=800.0)
households = st.number_input("Households", min_value=0.0, value=300.0)
housing_median_age = st.number_input("Housing Median Age", min_value=0.0, value=25.0)
longitude = st.number_input("Longitude", min_value=-125.0, max_value=-113.0, value=-120.0)
latitude = st.number_input("Latitude", min_value=32.0, max_value=42.0, value=36.0)
ocean_proximity = st.selectbox("Ocean Proximity", ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"])

if st.button("Predict"):
    # Manual feature engineering (same as training)
    rooms_per_household = total_rooms / households
    bedrooms_per_room = total_bedrooms / total_rooms
    population_per_household = population / households

    # Encode ocean proximity manually (same order as training)
    categories = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    ocean_encoded = [1 if ocean_proximity == cat else 0 for cat in categories]

    # Create final input array
    input_data = np.array([[longitude, latitude, housing_median_age, total_rooms,
                            total_bedrooms, population, households, median_income,
                            rooms_per_household, bedrooms_per_room, population_per_household] + ocean_encoded])

    # Make prediction
    prediction = model.predict(input_data)
    st.success(f"Predicted Median House Value: ${prediction[0]:,.2f}")
