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

st.set_page_config(page_title="Show Up for Health", page_icon="🩺")


# Home page function
def home_page():
    # Get the current working directory
    current_directory = os.getcwd()

    # Define the path to Jan's image
    relative_image_path = "images/Show_Up_for_Health.png"
    image_path = os.path.join(current_directory, relative_image_path)

    # Display home page image
    st.image(image_path, use_column_width=True)

    st.write("")
    st.write("")

    # Create a file uploader widget
    uploaded_file = st.file_uploader(
        "Upload a file", type=["csv", "txt", "xlsx", "json"]
    )

    st.write("")
    st.write("")

    # Drop down menu for clinics
    surgery_prefix = st.selectbox(
        "Which clinic would you like to select?",
        ("ECS", "HPVM", "KMC", "SMW", "TCP", "TGP"),
    )

    # Print selected clinic
    st.write("You selected:", surgery_prefix)

    # Check if a file has been uploaded
    if uploaded_file is not None:
        st.success("File uploaded successfully!")

        # You can perform operations on the uploaded file here
        # For example, you can read and display the content of the file:
        # file_contents = uploaded_file.read()
        # st.write("File Contents:")
        # st.write(file_contents.decode())

        # Parse the CSV file into a pandas DataFrame
        try:
            df = pd.read_csv(uploaded_file)  # Assuming it's a CSV file
            # st.write("Data Preview:")
            st.write(df.head())  # Display the first few rows of the DataFrame

            # Take uploaded_file, surgery_prefix, and df
            # Perform your model operations here and display the results
            # For example:
            # result = run_your_model(df, surgery_prefix)
            # st.write("Model Results:")
            # st.write(result)

        except Exception as e:
            st.error(
                "Error reading the uploaded file. Please make sure it's in the correct format."
            )
            st.exception(e)

    else:
        st.write("")

    st.write("")

    # Create a button, when clicked, run prediction
    if st.button("Predict"):
        # Use model to predict (you can add your model code here)
        # For example, assuming you have a function run_your_model()
        # prediction = run_your_model(df, surgery_prefix)
        prediction = final_predict(
            "HPVM"
        )  # Replace None with your actual prediction result
        public_predict = prediction[
            ["Model_Prediction", "Appointment date", "Appointment time", "Clinician"]
        ]
        # Display prediction
        st.dataframe(public_predict)


# About page function
def about_page():
    # Get the current working directory
    current_directory = os.getcwd()

    # Define the path to Jan's image
    relative_image_path_jan = "images/jancrop_bw.png"
    image_path_jan = os.path.join(current_directory, relative_image_path_jan)

    # Define the path to Michael's image
    relative_image_path_michael = "images/michaelcrop_bw.png"
    image_path_michael = os.path.join(current_directory, relative_image_path_michael)

    # Define the path to Alessio's image
    relative_image_path_alessio = "images/Alex_BW.png"
    image_path_alessio = os.path.join(current_directory, relative_image_path_alessio)

    # Define the path to Fabio's image
    relative_image_path_fabio = "images/Fabiocrop_bw.png"
    image_path_fabio = os.path.join(current_directory, relative_image_path_fabio)

    st.header("About the Project")

    st.subheader("Our Inspiration:")
    st.write(
        "The NHS faces an annual cost of approximately £1 billion due to missed appointments, commonly known as DNAs (Did Not Attends). These DNAs not only drain resources but also extend wait times for other patients who could have used those slots. With a staggering 4% DNA rate among 140,000 patients in our primary care network alone, we decided it was time for a change."
    )

    st.subheader("The Problem We're Solving :")
    st.write(
        "While the issue of DNAs has gained some attention, most existing predictive models are focused on secondary care. This leaves primary care, the frontline of healthcare, underrepresented in data-driven solutions. Moreover, telephone appointments are often not counted, concealing the true scale of the issue."
    )

    st.subheader("Our App:")
    st.write(
        "Our app harnesses the power of deep learning to predict the likelihood of DNAs in primary care appointments. We're not just looking at past attendance records; we're also integrating variables like health indicators, local weather conditions, index of multiple deprivation and more, to give healthcare providers a more holistic understanding of patient behaviour."
    )

    st.write("")
    st.write("")

    st.subheader("The Team:")

    # Create columns for images and names side by side
    col1, col2, col3, col4 = st.columns(4)

    # Display Jan's image and name
    with col1:
        st.image(image_path_jan, width=150)
        st.write("Jan du Plessis")

    # Display Michael's image and name
    with col2:
        st.image(image_path_michael, width=150)
        st.write("Michael Melis")

    # Display Alessio's image and name
    with col3:
        st.image(image_path_alessio, width=150)
        st.write("Alessio Robotti")

    # Display Fabio's image and name
    with col4:
        st.image(image_path_fabio, width=150)
        st.write("Fabio Sparano")


# Main function
def main():
    # st.title('Show up for Health')
    # Create a sidebar menu
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox("Select a page", ["Home", "About"])
    # Depending on the selected page, display content
    if page == "Home":
        home_page()
    elif page == "About":
        about_page()


if __name__ == "__main__":
    main()


# df = pd.read_csv(f"{PREDICT_DATA}/original/{surgery_prefix}_Predict.csv")
# df = pd.DataFrame(
# data={
# "Appointment status": {
# 8: "Finished",
# 214: "Finished",
# 225: "Finished",
# 422: "Finished",
# 423: "Did Not Attend",
# 425: "Did Not Attend",
# 474: "Did Not Attend",
# 475: "Did Not Attend",
# 476: "Did Not Attend",
# 477: "Did Not Attend",
# 590: "Finished",
# 618: "Did Not Attend",
# 1098: "Finished",
# 1165: "Finished",
# 1310: "Finished",
# 1626: "Finished",
# 1627: "Finished",
# 1628: "Finished",
# 1825: "Did Not Attend",
# 1826: "Did Not Attend",
# 1855: "Finished",
# 1856: "Finished",
# 1857: "Finished",
# 2090: "Finished",
# 2178: "Finished",
# 2179: "Finished",
# 2213: "Finished",
# 2214: "Finished",
# 2325: "Finished",
# 2335: "Finished",
# 2336: "Finished",
# 2404: "Finished",
#             2405: "Finished",
#             2931: "Finished",
#             2932: "Finished",
#             3648: "Finished",
#             3731: "Finished",
#             3867: "Finished",
#             3868: "Finished",
#             4137: "Finished",
#             4139: "Finished",
#             4420: "Did Not Attend",
#             4421: "Did Not Attend",
#             4423: "Did Not Attend",
#             4447: "Finished",
#             4448: "Finished",
#             4449: "Finished",
#             4795: "Finished",
#             4902: "Finished",
#             4962: "Did Not Attend",
#             4988: "Finished",
#             4990: "Finished",
#             5002: "Did Not Attend",
#             5053: "Finished",
#             5475: "Finished",
#             5849: "Finished",
#             6046: "Finished",
#             6122: "Finished",
#             6123: "Finished",
#             6124: "Finished",
#             6170: "Finished",
#             6311: "Finished",
#             6359: "Finished",
#             6370: "Finished",
#             6416: "Finished",
#             6459: "Finished",
#             6486: "Finished",
#             6547: "Finished",
#             6599: "Did Not Attend",
#             6653: "Did Not Attend",
#             6657: "In Progress",
#             6659: "In Progress",
#             6775: "Finished",
#             6980: "Did Not Attend",
#             7706: "Finished",
#             7820: "Finished",
#             7821: "Finished",
#             7947: "Finished",
#             7948: "Finished",
#             7949: "Finished",
#             8021: "Finished",
#             8022: "Finished",
#             8023: "Finished",
#             8257: "Finished",
#             8293: "Finished",
#             8361: "Finished",
#         },
#         "Model_Prediction": {
#             8: 0.0,
#             214: 0.0,
#             225: 0.0,
#             422: 0.0,
#             423: 0.0,
#             425: 0.0,
#             474: 0.0,
#             475: 0.0,
#             476: 0.0,
#             477: 0.0,
#             590: 0.0,
#             618: 0.0,
#             1098: 0.0,
#             1165: 0.0,
#             1310: 0.0,
#             1626: 0.0,
#             1627: 0.0,
#             1628: 0.0,
#             1825: 0.0,
#             1826: 0.0,
#             1855: 0.0,
#             1856: 0.0,
#             1857: 0.0,
#             2090: 0.0,
#             2178: 0.0,
#             2179: 0.0,
#             2213: 0.0,
#             2214: 0.0,
#             2325: 0.0,
#             2335: 0.0,
#             2336: 0.0,
#             2404: 0.0,
#             2405: 0.0,
#             2931: 0.0,
#             2932: 0.0,
#             3648: 0.0,
#             3731: 0.0,
#             3867: 0.0,
#             3868: 0.0,
#             4137: 0.0,
#             4139: 0.0,
#             4420: 0.0,
#             4421: 0.0,
#             4423: 0.0,
#             4447: 0.0,
#             4448: 0.0,
#             4449: 0.0,
#             4795: 0.0,
#             4902: 0.0,
#             4962: 0.0,
#             4988: 0.0,
#             4990: 0.0,
#             5002: 0.0,
#             5053: 0.0,
#             5475: 0.0,
#             5849: 0.0,
#             6046: 0.0,
#             6122: 0.0,
#             6123: 0.0,
#             6124: 0.0,
#             6170: 0.0,
#             6311: 0.0,
#             6359: 0.0,
#             6370: 0.0,
#             6416: 0.0,
#             6459: 0.0,
#             6486: 0.0,
#             6547: 0.0,
#             6599: 0.0,
#             6653: 0.0,
#             6657: 0.0,
#             6659: 0.0,
#             6775: 0.0,
#             6980: 0.0,
#             7706: 0.0,
#             7820: 0.0,
#             7821: 0.0,
#             7947: 0.0,
#             7948: 0.0,
#             7949: 0.0,
#             8021: 0.0,
#             8022: 0.0,
#             8023: 0.0,
#             8257: 0.0,
#             8293: 0.0,
#             8361: 0.0,
#         },
#     }
# )


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
