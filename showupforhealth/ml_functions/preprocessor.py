# Import Libriries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import hashlib
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import time

from showupforhealth.params import *
from showupforhealth.ml_functions.encoders import extract_rota_type


# Create a new column 'Booked_by_Gp' with 1 if booked by the same clinician, else 0
def booked_by_clinicain(df):
    print("➡️ Encoded Booked by Clinician")
    df["booked_by_clinician"] = df.apply(
        lambda row: 1 if row["Clinician"] == row["Booked by"] else 0, axis=1
    )
    df.drop(
        columns=["Booked by", "Clinician", "NHS number", "app datetime"], inplace=True
    )
    return df


def hash_patient_id(df, length=8):
    print("➡️ Hash Patient ID")
    df["Patient ID"] = df["Patient ID"].apply(
        lambda x: hashlib.sha512(str(x).encode("utf-8")).hexdigest()[:length]
    )
    return df


def one_hot_encode_columns(df, columns_to_encode=["Rota"]):
    print("➡️ OHE Columns")
    """
    Perform One-Hot Encoding on specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (list): List of column names to One-Hot Encode.

    Returns:
        pd.DataFrame: A new DataFrame with One-Hot Encoded columns and original columns dropped.
    """
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # Fit the encoder to the specified categorical columns and transform the data
    encoded_data = encoder.fit_transform(df[columns_to_encode])

    # Convert the result to a DataFrame
    encoded_df = pd.DataFrame(
        encoded_data.toarray(), columns=encoder.get_feature_names_out(columns_to_encode)
    )

    # Concatenate the encoded DataFrame with the original DataFrame, dropping the original columns
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=columns_to_encode, inplace=True)
    return df


def encode_appointment_status(df):
    print("➡️ Encode Appointment status")
    df["Appointment_status"] = [
        0 if app_status == "Did Not Attend" else 1
        for app_status in df["Appointment status"]
    ]
    df = df.drop(columns="Appointment status")
    return df


def encode_hour_appointment(df):
    print("➡️ Extract Time of Appointment")
    df["hour_of_appointment"] = df["Appointment time"].str[:2].astype(int)
    df.drop(columns=["Appointment time"], inplace=True)
    return df


def group_ethnicity_categories(df):
    print("➡️ Mapping Ethnicity category")
    ethnicity_dict = {
        "African": "Black",
        "Other Black": "Black",
        "Caribbean": "Black",
        "British or Mixed British": "White",
        "Other White": "White",
        "Irish": "White",
        "White & Black African": "Mixed",
        "White & Black Caribbean": "Mixed",
        "White & Asian": "Mixed",
        "Other Mixed": "Mixed",
        "Other Asian": "Asian",
        "Indian or British Indian": "Asian",
        "Pakistani or British Pakistani": "Asian",
        "Chinese": "Asian",
        "Bangladeshi or British Bangladeshi": "Asian",
        "Other": "Unknown",
    }
    df["Ethnicity category"] = df["Ethnicity category"].map(ethnicity_dict)
    return df


# Jan pre-processor Functions
def format_datetime_columms(df):
    print("➡️ Convert Datetime Columns")
    date_list = ["Appointment booked date", "Appointment date", "Registration date"]
    for date in date_list:
        df[date] = pd.to_datetime(df[date])
    return df


def months_registered(df):
    print("➡️ Calculate months registered with practice")
    # Calculate the difference between two dates
    df["delta"] = df["Appointment date"] - df["Registration date"]

    # Convert Timedelta to months
    df["months_registered"] = df["delta"].dt.total_seconds() / (60 * 60 * 24 * 30.44)
    df["months_registered"] = np.ceil(df["months_registered"])

    # Drop the temporary 'delta' column
    df.drop(columns=["delta"], inplace=True)
    return df


def map_rota_type(df):
    print("➡️ Map Rota types - renamed Rota")
    df["Rota type"] = df["Rota type"].map(extract_rota_type)

    df.drop(df[df["Rota type"] == "DROP"].index, inplace=True)
    # Rename column
    df.rename(columns={"Rota type": "Rota"}, inplace=True)
    return df


# Label Encode Sex
def labelencode_sex(df):
    print("➡️ Labelencoding Sex")
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])
    return df


def split_appointment_date(df):
    print("➡️ Split appointment date: week, months, day_of_week")
    # Convert the "Appointment date" column to datetime if it's not already
    df["Appointment date"] = pd.to_datetime(df["Appointment date"], format="%d-%b-%y")

    # Extract day, week, and month
    df["Day"] = df["Appointment date"].dt.dayofweek  # 0-6 (Monday to Sunday)
    df["Week"] = df["Appointment date"].dt.isocalendar().week  # 1-52
    df["Month"] = df["Appointment date"].dt.month  # 1-12 (January to December)
    return df


def filter_current_registration(df):
    print("➡️ Drop deseased and deducted")
    # Filter rows where 'Registration status' is 'Current'
    df.drop(df[df["Registration status"] == "Deceased, Deducted"].index, inplace=True)
    df.drop(df[df["Registration status"] == "Deceased"].index, inplace=True)
    df.drop(df[df["Registration status"] == "Deducted"].index, inplace=True)
    return df


# days until appointment function
def calculate_days_difference(df):
    print("➡️ Calculate how long before appointment date booked")
    # Convert the date columns to datetime objects
    df["Appointment booked date"] = pd.to_datetime(
        df["Appointment booked date"], format="%d-%b-%y"
    )
    df["Appointment date"] = pd.to_datetime(df["Appointment date"], format="%d-%b-%y")

    # Calculate the difference in days and create a new column
    df["days_booked_to_app"] = (
        df["Appointment date"] - df["Appointment booked date"]
    ).dt.days

    # Drop rows where 'Days Difference' is negative
    df = df[df["days_booked_to_app"] >= 0]
    return df


def drop_rename_columns(df):
    print("➡️ Drop and rename columns")
    df.drop(
        columns=[
            "Postcode",
            "Latitude",
            "Longitude",
            "Appointment booked date",
            "Appointment status",
            "Registration date",
            "Registration status",
        ],
        inplace=True,
    )
    df.rename(columns={"Age in years": "Age"}, inplace=True)
    return df


def feature_engeneering(df):
    start_time = time.time()
    print(
        "\n==== Feature Engineering ====================================================="
    )
    print(f"👉 Input shape {df.shape}")
    format_datetime_columms(df)
    booked_by_clinicain(df)
    hash_patient_id(df)
    encode_appointment_status(df)
    group_ethnicity_categories(df)
    months_registered(df)
    map_rota_type(df)
    labelencode_sex(df)
    encode_hour_appointment(df)
    split_appointment_date(df)
    filter_current_registration(df)
    calculate_days_difference(df)
    one_hot_encode_columns(df)
    drop_rename_columns(df)
    print("💾 Saving to output_data/full_train_data.csv...")
    df.to_csv(f"{OUTPUT_DATA}full_train_data.csv", index=False)
    end_time = time.time()
    print(f"✅ Done in {round((end_time - start_time),2)} sec {df.shape}")
    return df
