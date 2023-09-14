import streamlit as st
import pandas as pd
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
    time.sleep(0.08)
    my_bar.progress(percent_complete + 1)

st.success(success_text)

# Turning the uploaded file into a dataframe
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write(df)


# Creating a function for the prediction of patients showing up or not to thier appointments

# def show_up(uploaded_file):

#     if prediction == 1:
#         st.write("The patient will show up to their appointment")
#     else:
#         st.write("The patient will not show up to their appointment")
