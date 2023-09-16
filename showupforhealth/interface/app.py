import streamlit as st
import pandas as pd
import numpy as np
import time
import os
from showupforhealth.ml_functions.predict import *

# Page title
st.title("Show up for Health")

# Drop down menu for clinics
clinic_option = st.selectbox(
    "Which clinic would you like to select?",
    ("ECS", "HPVM", "KMC", "SMW", "TCP", "TGP"),
)

st.write("You selected:", clinic_option)

# Define the directory where your CSV files are stored
data_directory = "/Users/fabiosparano/code/janduplessis883/data-showup/data/predict"

# Construct the file path based on the selected clinic
file_path = os.path.join(data_directory, f"{clinic_option}_predict.csv")

# st.write(file_path)

# Turning the uploaded file into a dataframe
df = pd.read_csv(file_path)
st.write(df)

def predict_add_weather(df):
    print(
        "\n=== Preparing Appoitment Data for Prediction ================================="
    )
    # print(f"🌤️ Prediction: {surgery_prefix} - preparing appointment data for weather")
    # data = pd.read_csv(f"{PREDICT_DATA}/{surgery_prefix}_predict.csv")
    # print(
    #     f"👩🏻‍🦰 Appointments: {data.shape[0]} 🧑🏻‍🦰 Unique Patient IDs; {data['Patient ID'].nunique()}"
    # )
    df["weather_time"] = df["Appointment time"].str.split("-").str[0]
    df["weather_datetime"] = df["Appointment date"] + " " + df["weather_time"]
    df["Appointment date"] = pd.to_datetime(df["Appointment date"])
    df["weather_datetime"] = pd.to_datetime(df["weather_datetime"])
    start_date = df["Appointment date"].dt.strftime("%Y-%m-%d").min()
    end_date = df["Appointment date"].dt.strftime("%Y-%m-%d").max()

    # Getting API Data
    # Define the base URL of the API
    base_url = "https://api.open-meteo.com/v1/forecast"

    # Define the parameters as a dictionary
    params = {
        "latitude": "51.5085",
        "longitude": "0.1257",
        "hourly": "temperature_2m,precipitation",
        "start_date": start_date,
        "end_date": end_date
        # Add more parameters as needed
    }

    # Make the API call using the requests library
    response = requests.get(base_url, params=params)
    print(f"🛜 Requesting forcast from Open-Meteo Weather API {start_date} - {end_date}")
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse and work with the API response, which is typically in JSON format
        api_data = response.json()["hourly"]
        # Now you can work with the data returned by the API
        api_data = pd.DataFrame(api_data)
    else:
        # If the request was not successful, handle the error
        print(f"Error: {response.status_code}")
        print(response.text)

    api_data = api_data.rename(
        columns={
            "time": "weather_datetime",
            "temperature_2m": "temp",
            "precipitation": "precipitation",
        }
    )
    api_data["weather_datetime"] = pd.to_datetime(api_data["weather_datetime"])

    # Merging dataframes
    print(f"🔂 Merge weather + appointment data")
    df = df.merge(api_data, how="left", on="weather_datetime")
    nanindf = df.isna().sum()

    if nanindf.sum() > 0:
        print(f"❌ NaN values in df - ERROR: {nanindf}")
    else:
        print("✅ Successful: return df")
        df["Patient ID"] = df["Patient ID"].astype("int")
    return df


st.write(predict_add_weather(df))

st.write(predict_add_global_register(df))

st.write(add_noshows(df))

st.write(predict_feature_engineering(df))




# Create a button, when clicked, run prediction

# Load the trained model (replace with your model file)
# model = model_name()
# model.load('our_model.pkl')

# # Create input fields for user to input data
# feature1 = st.number_input('Input feature 1')
# feature2 = st.number_input('Input feature 2')
# feature3 = st.number_input('Input feature 3')


st.button("Predict")
#     # Reshape inputs to match model's input shape
#     data = np.array([feature1, feature2, feature3]).reshape(1, -1)

#     # Use model to predict
#     prediction = model.predict(data)

#     # Display prediction
#     st.write(f'Prediction: {prediction}')


# Button to download the dataframe as a csv file
# @st.cache
# def convert_df(df):
#     # IMPORTANT: Cache the conversion to prevent computation on every rerun
#     return df.to_csv().encode('utf-8')

# csv = convert_df(df)

# st.download_button(
#     label="Download data as CSV",
#     data=csv,
#     file_name='large_df.csv',
#     mime='text/csv',
# )


# Creating a function for the prediction of patients showing up or not to thier appointments

# def show_up(uploaded_file):

#     if prediction == 1:
#         st.write("The patient will show up to their appointment")
#     else:
#         st.write("The patient will not show up to their appointment")
