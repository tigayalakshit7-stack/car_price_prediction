# Sreamlit App for Car Price Prediction

#Step 1 : Import libraries

import streamlit as st
import pandas as pd
import joblib

#Step 2 : Load saved model and Features

model = joblib.load("car_price_model.pkl")
model_features = joblib.load("model_features.pkl")

#Step 3 : App Title

st.title("Car Price Prediction using Random Forest")

st.write("Enter car details to predict selling price.")

#Step 4 : User Inputs

present_price = st.number_input(
    "Present Price of Car in Lakhs",
    min_value=0.0,
    max_value=100.0,
    value=5.0
    )

kms_driven = st.number_input(
    "kilometers Driven",
    min_value=0,
    max_value=500000,
    value=30000
    )

owner = st.selectbox(
    "Number of Previous Owners",
    [0,1,2,3]
    )

car_age = st.number_input(
    "Car Age",
    min_value=0,
    max_value=30,
    value=5
    )

fuel_type = st.selectbox(
    "Fuel Type",
    ["Petrol","diesel","CNG"]
    )

seller_type = st.selectbox(
    "Seller Type",
    ["Dealer","Individual"]
    )

transmission = st.selectbox(
    "Transmission",
    ["Manual","Automatic"]
    )

#Step 5 : Create Input dictionary

input_data = {
    "Present_Price": present_price,
    "Kms_Driven": kms_driven,
    "Owner": owner,
    "Car_Age": car_age
    }

#Step 6 : Convert user Input into dataframe

input_df = pd.DataFrame([input_data])

#Step 7 : Add Categorical dummy columns manually

input_df["Fuel_Type_Diesel"] = 1 if fuel_type == "Diesel" else 0
input_df["Fuel_Type_Petrol"] = 1 if fuel_type == "Petrol" else 0

input_df["Seller_Type_Individual"] = 1 if seller_type == "Individual" else 0
input_df["Transmission_Manual"] = 1 if transmission == "Manual" else 0

#Step 8 : Match input columns with Training columns

for col in model_features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[model_features]

#Step 9 : Prediction button

if st.button("Prediction Car Price"):
    prediction = model.predict(input_df)

    st.success(f"Estimated Selling Price : ₹ {prediction[0]:.2f} Lakhs")
