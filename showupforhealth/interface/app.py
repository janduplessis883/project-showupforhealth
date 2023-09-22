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
from showupforhealth.ml_functions.model import *
from showupforhealth.interface.model_predict import *

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

# df = pd.read_csv(f"{PREDICT_DATA}/original/{surgery_prefix}_Predict.csv")
df = pd.DataFrame(
    data={
        "Appointment status": {
            8: "Finished",
            214: "Finished",
            225: "Finished",
            422: "Finished",
            423: "Did Not Attend",
            425: "Did Not Attend",
            474: "Did Not Attend",
            475: "Did Not Attend",
            476: "Did Not Attend",
            477: "Did Not Attend",
            590: "Finished",
            618: "Did Not Attend",
            1098: "Finished",
            1165: "Finished",
            1310: "Finished",
            1626: "Finished",
            1627: "Finished",
            1628: "Finished",
            1825: "Did Not Attend",
            1826: "Did Not Attend",
            1855: "Finished",
            1856: "Finished",
            1857: "Finished",
            2090: "Finished",
            2178: "Finished",
            2179: "Finished",
            2213: "Finished",
            2214: "Finished",
            2325: "Finished",
            2335: "Finished",
            2336: "Finished",
            2404: "Finished",
            2405: "Finished",
            2931: "Finished",
            2932: "Finished",
            3648: "Finished",
            3731: "Finished",
            3867: "Finished",
            3868: "Finished",
            4137: "Finished",
            4139: "Finished",
            4420: "Did Not Attend",
            4421: "Did Not Attend",
            4423: "Did Not Attend",
            4447: "Finished",
            4448: "Finished",
            4449: "Finished",
            4795: "Finished",
            4902: "Finished",
            4962: "Did Not Attend",
            4988: "Finished",
            4990: "Finished",
            5002: "Did Not Attend",
            5053: "Finished",
            5475: "Finished",
            5849: "Finished",
            6046: "Finished",
            6122: "Finished",
            6123: "Finished",
            6124: "Finished",
            6170: "Finished",
            6311: "Finished",
            6359: "Finished",
            6370: "Finished",
            6416: "Finished",
            6459: "Finished",
            6486: "Finished",
            6547: "Finished",
            6599: "Did Not Attend",
            6653: "Did Not Attend",
            6657: "In Progress",
            6659: "In Progress",
            6775: "Finished",
            6980: "Did Not Attend",
            7706: "Finished",
            7820: "Finished",
            7821: "Finished",
            7947: "Finished",
            7948: "Finished",
            7949: "Finished",
            8021: "Finished",
            8022: "Finished",
            8023: "Finished",
            8257: "Finished",
            8293: "Finished",
            8361: "Finished",
        },
        "Model_Prediction": {
            8: 0.0,
            214: 0.0,
            225: 0.0,
            422: 0.0,
            423: 0.0,
            425: 0.0,
            474: 0.0,
            475: 0.0,
            476: 0.0,
            477: 0.0,
            590: 0.0,
            618: 0.0,
            1098: 0.0,
            1165: 0.0,
            1310: 0.0,
            1626: 0.0,
            1627: 0.0,
            1628: 0.0,
            1825: 0.0,
            1826: 0.0,
            1855: 0.0,
            1856: 0.0,
            1857: 0.0,
            2090: 0.0,
            2178: 0.0,
            2179: 0.0,
            2213: 0.0,
            2214: 0.0,
            2325: 0.0,
            2335: 0.0,
            2336: 0.0,
            2404: 0.0,
            2405: 0.0,
            2931: 0.0,
            2932: 0.0,
            3648: 0.0,
            3731: 0.0,
            3867: 0.0,
            3868: 0.0,
            4137: 0.0,
            4139: 0.0,
            4420: 0.0,
            4421: 0.0,
            4423: 0.0,
            4447: 0.0,
            4448: 0.0,
            4449: 0.0,
            4795: 0.0,
            4902: 0.0,
            4962: 0.0,
            4988: 0.0,
            4990: 0.0,
            5002: 0.0,
            5053: 0.0,
            5475: 0.0,
            5849: 0.0,
            6046: 0.0,
            6122: 0.0,
            6123: 0.0,
            6124: 0.0,
            6170: 0.0,
            6311: 0.0,
            6359: 0.0,
            6370: 0.0,
            6416: 0.0,
            6459: 0.0,
            6486: 0.0,
            6547: 0.0,
            6599: 0.0,
            6653: 0.0,
            6657: 0.0,
            6659: 0.0,
            6775: 0.0,
            6980: 0.0,
            7706: 0.0,
            7820: 0.0,
            7821: 0.0,
            7947: 0.0,
            7948: 0.0,
            7949: 0.0,
            8021: 0.0,
            8022: 0.0,
            8023: 0.0,
            8257: 0.0,
            8293: 0.0,
            8361: 0.0,
        },
    }
)

# Create a button, when clicked, run prediction
if st.button("Predict"):
    # Use model to predict
    prediction = st.dataframe(df)

    # Display prediction
    st.write(f"Prediction: {prediction}")


# Load the trained model (replace with your model file)
# model = model_name()
# model.load('our_model.pkl')

# # Create input fields for user to input data
# feature1 = st.number_input('Input feature 1')
# feature2 = st.number_input('Input feature 2')
# feature3 = st.number_input('Input feature 3')


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
