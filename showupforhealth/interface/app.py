import streamlit as st
import pandas as pd
import numpy as np
import time

# Page title
st.title("Show up for Health")

# Uploaded banner
uploaded_file = st.file_uploader("Upload your file here")

# Cretaing a progress bar whilst the file is being uploaded


# Create a progress bar
my_bar = st.progress(0)

success_text = "File uploaded successfully!"

for percent_complete in range(100):
     time.sleep(0.05)
     my_bar.progress(percent_complete + 1)

st.success(success_text)

# Turning the uploaded file into a dataframe
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df)


st.button('Predict')

# Load your trained model (replace with your model file)
# model = model.name()
# model.load('your_model.pkl ???predict.csv???')

# # Create input fields for user to input data
# feature1 = st.number_input('Input feature 1')
# feature2 = st.number_input('Input feature 2')
# feature3 = st.number_input('Input feature 3')

# Create a button, when clicked, run prediction
# if st.button('Predict'):
    # Reshape inputs to match model's input shape
    # df = np.array([feature1, feature2, feature3]).reshape(1, -1)

    # # Use model to predict
    # prediction = model.predict(df)

    # # Display prediction
    # st.write(f'Prediction: {prediction}')






# Creating a function for the prediction of patients showing up or not to thier appointments

# def show_up(uploaded_file):

#     if prediction == 1:
#         st.write("The patient will show up to their appointment")
#     else:
#         st.write("The patient will not show up to their appointment")
