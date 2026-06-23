import streamlit as st
import pickle
import gzip
import numpy as np

with gzip.open('house_price_model.pkl.gz', 'rb') as f:
    model = pickle.load(f)

st.title(" House Price Prediction")
st.write("Enter the house details below to predict the price!")

MedInc = st.slider("Median Income (in $10,000s)", 0.5, 15.0, 3.0)
HouseAge = st.slider("House Age (years)", 1, 52, 20)
AveRooms = st.slider("Average Rooms", 1.0, 20.0, 5.0)
AveBedrms = st.slider("Average Bedrooms", 0.5, 5.0, 1.0)
Population = st.number_input("Population in Area", min_value=1, max_value=40000, value=1000)
AveOccup = st.slider("Average Occupants per Household", 1.0, 10.0, 2.5)
Latitude = st.slider("Latitude", 32.5, 42.0, 37.0)
Longitude = st.slider("Longitude", -124.5, -114.0, -119.0)

if st.button("Predict Price"):
    features = np.array([[MedInc, HouseAge, AveRooms, AveBedrms,
                          Population, AveOccup, Latitude, Longitude]])
    
    prediction = model.predict(features)[0]
    st.success(f"Estimated House Price: **${prediction * 100000:,.0f}**")
    st.info(f"(Model predicts: {prediction:.4f} × $100,000)")