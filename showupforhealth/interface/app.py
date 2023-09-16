import streamlit as st
import pandas as pd
import numpy as np
import time

# Page title
st.title("Show up for Health")

# Drop down menu for clinics
clinic_option = st.selectbox(
    "Which clinic would you like to select?",
    ("ECS", "HPVM", "KMC", "SMW", "TCP", "TGP"),
)

st.write("You selected:", clinic_option)

# Uploaded banner
uploaded_file = st.file_uploader("Upload your file here")

# Creating a progress bar whilst the file is being uploaded
my_bar = st.progress(0)

success_text = "File uploaded successfully!"

for percent_complete in range(100):
    time.sleep(0.08)
    my_bar.progress(percent_complete + 1)

st.success(success_text)

# Turning the uploaded file into a dataframe
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df)


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
