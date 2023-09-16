import streamlit as st
import pandas as pd
import numpy as np

import time
import os

from showupforhealth.ml_functions import *
from showupforhealth.ml_functions.predict import *
from showupforhealth.params import *
from showupforhealth.utils import *
from showupforhealth.ml_functions.encoders import *

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


# Page title
st.title("Show up for Health")

# Drop down menu for clinics
surgery_prefix = st.selectbox(
    "Which clinic would you like to select?",
    ("ECS", "HPVM", "KMC", "SMW", "TCP", "TGP"),
)

# Print selected clinic
st.write("You selected:", surgery_prefix)

# Show dataframe of predictions
st.write(streamlit_predict(surgery_prefix))




  

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
#     prediction = model.predict(df)

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
