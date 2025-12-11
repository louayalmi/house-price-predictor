import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open("house_model.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

qual_map = {"Ex":5, "Gd":4, "TA":3, "Fa":2, "Po":1}

st.title("House Price Predictor")
st.markdown("Built today in a few hours • XGBoost • R² ≈ 0.92")

col1, col2 = st.columns(2)
with col1:
    qual = st.slider("Overall Quality", 1, 10, 7)
    sqft = st.slider("Total House Area (sqft)", 800, 6000, 2000)
    garage = st.selectbox("Garage spaces", [0,1,2,3,4], index=2)
    baths = st.slider("Total bathrooms", 1.0, 5.0, 2.5, 0.5)

with col2:
    age = st.slider("House age (years)", 0, 150, 35)
    neighborhood = st.slider("Neighborhood quality", 80_000, 300_000, 170_000)
    kitchen = st.selectbox("Kitchen quality", list(qual_map.keys()), 2)
    exterior = st.selectbox("Exterior quality", list(qual_map.keys()), 2)

if st.button("Predict Price"):
    input_dict = {col: 0 for col in features}
    input_dict.update({
        'OverallQual': qual,
        'TotalSF': sqft,
        'GarageCars': garage,
        'TotalBath': baths,
        'HouseAge': age,
        'Neighborhood': neighborhood,
        'KitchenQual': qual_map[kitchen],
        'ExterQual': qual_map[exterior],
        'GrLivArea': sqft * 0.7,
        'GarageArea': garage * 500,
        'YearBuilt': 2025 - age,
        'YearRemodAdd': 2025 - age + 5,
    })
    X_input = pd.DataFrame([input_dict])[features]

    price = np.expm1(model.predict(X_input)[0])
    st.balloons()
    st.success(f"Predicted price: **${price:,.0f}**")
