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
from showupforhealth.ml_functions.encoders import *


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


import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def encode_ethnicity_categories(df):
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

    if "Ethnicity category" not in df.columns:
        raise ValueError("Ethnicity category column not found in DataFrame")

    df["Ethnicity category"] = df["Ethnicity category"].map(ethnicity_dict)
    df.rename(columns={"Ethnicity category": "Ethnicity"}, inplace=True)

    # Initialize OneHotEncoder
    encoder = OneHotEncoder(categories="auto", sparse=False)

    # Fit and transform the 'Ethnicity' column
    encoded_ethnicity = encoder.fit_transform(df[["Ethnicity"]])

    # Convert the encoded array back into a DataFrame
    encoded_ethnicity_df = pd.DataFrame(
        encoded_ethnicity, columns=encoder.get_feature_names_out(["Ethnicity"])
    )

    # Concatenate the original DataFrame and the encoded DataFrame
    data = pd.concat([df, encoded_ethnicity_df], axis=1)

    # Drop the original 'Ethnicity' column
    data = data.drop(["Ethnicity"], axis=1)

    return data


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


def one_hot_encode_columns(df, columns_to_encode=["Rota", "Ethnicity"]):
    print("➡️ OHE Columns Rota & Ethnicity")

    # Print the original DataFrame
    print("Original DataFrame:")
    print(df.head())

    df = df.copy()
    encoder = OneHotEncoder(categories="auto", sparse=False)
    encoded_dfs = []

    for column in columns_to_encode:
        # Make sure the column is in the DataFrame
        if column not in df.columns:
            print(f"Column {column} not found in DataFrame.")
            continue

        encoded = encoder.fit_transform(df[[column]])

        # Print the encoded column
        print(f"Encoded {column}:")
        print(encoded)

        feature_names = [f"{column}_{category}" for category in encoder.categories_[0]]

        # Print feature names
        print(f"Feature names for {column}:")
        print(feature_names)

        encoded_data = pd.DataFrame(encoded, columns=feature_names, index=df.index)
        encoded_dfs.append(encoded_data)

    new_df = pd.concat(encoded_dfs, axis=1)
    new_df = df.drop(columns_to_encode, axis=1)

    # Print the final DataFrame
    print("Final DataFrame:")
    print(new_df.head())

    return new_df


def feature_engineering(df):
    start_time = time.time()
    print(
        "=== Feature Engineering ============================================================="
    )
    print("🔂 Rename Columns")
    df.rename(
        columns={
            "Appointment status": "Appointment_status",
            "Booked by": "Booked_by",
            "Appointment time": "Appointment_time",
            "Rota type": "Rota",
            "Age in years": "Age",
        },
        inplace=True,
    )

    print("🔂 Drop deseased and deducted")
    # Filter rows where 'Registration status' is 'Current'
    df.drop(df[df["Registration status"] == "Deceased, Deducted"].index, inplace=True)
    df.drop(df[df["Registration status"] == "Deceased"].index, inplace=True)
    df.drop(df[df["Registration status"] == "Deducted"].index, inplace=True)

    # Convert Date Columns to DATETIME
    print("🔂 Columns to Datetime")
    datetime_cols = ["Appointment booked date", "Appointment date", "Registration date"]
    for datetime_col in datetime_cols:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    print("🔂 Fix Appointment Time")
    df["Appointment_time"] = df["Appointment_time"].astype("str")
    df["Appointment_time"] = df["Appointment_time"].str.split(":").str[0].astype(int)
    print("🔂 Map Appointment Status")
    df["Appointment_status"] = (
        df["Appointment_status"].map(fix_appointment_status).astype(int)
    )
    print("🔂 book_to_app_days")
    df["book_to_app_days"] = (
        df["Appointment date"] - df["Appointment booked date"]
    ).dt.total_seconds() / (60 * 60 * 24)

    # df['Appointment_time'] = df['Appointment_time'].astype('str')
    # df['Appointment_time'] = df['Appointment_time'].str.split(':').str[0].astype(int)
    print("🔂 booked_by_clinician")
    df["booked_by_clinician"] = (df["Booked_by"] == df["Clinician"]).astype(int)

    print("🔂 Extract Rota Types")
    df["Rota"] = df["Rota"].map(extract_rota_type)

    print("🔂 registered_for_months")
    df["registered_for_months"] = (
        (pd.Timestamp.now() - df["Registration date"]).dt.total_seconds()
        / (60 * 60 * 24 * 7 * 30)
    ).apply(np.ceil)

    # print('week month day_of_week')
    # df[['week', 'month', 'day_of_week']] = df['Appointment date'].apply(lambda x: pd.Series([x.week, x.month, x.dayofweek]))
    print("🔂 Week")
    df["week"] = df["Appointment date"].dt.isocalendar().week
    print("🔂 month")
    df["month"] = df["Appointment date"].dt.month
    print("🔂 day of week")
    df["day_of_week"] = df["Appointment date"].dt.dayofweek

    type_cast_cols = {
        "Appointment_time": "int",
        "Age": "int",
        "FRAILTY": "float",
        "DEPRESSION": "int",
        "OBESITY": "int",
        "IHD": "int",
        "DM": "int",
        "HPT": "int",
        "NDHG": "int",
        "SMI": "int",
    }
    for col, col_type in type_cast_cols.items():
        df[col] = df[col].astype(col_type)
    print("🔂 Convert Cyclical data")
    # Converting Weeks to Cyclical data
    cyclical_column = "week"
    weeks_in_a_year = 52
    df["sin_" + cyclical_column] = np.sin(
        2 * np.pi * df[cyclical_column] / weeks_in_a_year
    )
    df["cos_" + cyclical_column] = np.cos(
        2 * np.pi * df[cyclical_column] / weeks_in_a_year
    )
    df.drop(cyclical_column, axis=1, inplace=True)

    # Converting Appointment_time to Cyclical data
    cyclical_column = "Appointment_time"
    hrs_day = 24
    df["sin_" + cyclical_column] = np.sin(2 * np.pi * df[cyclical_column] / hrs_day)
    df["cos_" + cyclical_column] = np.cos(2 * np.pi * df[cyclical_column] / hrs_day)
    df.drop(cyclical_column, axis=1, inplace=True)

    # Convertingmonth to Cyclical data
    cyclical_column = "month"
    months_in_a_year = 12
    df["sin_" + cyclical_column] = np.sin(
        2 * np.pi * df[cyclical_column] / months_in_a_year
    )
    df["cos_" + cyclical_column] = np.cos(
        2 * np.pi * df[cyclical_column] / months_in_a_year
    )
    df.drop(cyclical_column, axis=1, inplace=True)

    cyclical_column = "day_of_week"
    day_per_week = 7
    df["sin_" + cyclical_column] = np.sin(
        2 * np.pi * df[cyclical_column] / day_per_week
    )
    df["cos_" + cyclical_column] = np.cos(
        2 * np.pi * df[cyclical_column] / day_per_week
    )
    df.drop(cyclical_column, axis=1, inplace=True)
    print("🔂 Adding NO Shows Column")
    # Filter to only rows where status = 0 (no-show)
    noshow = df[df["Appointment_status"] == 0]

    # Group by Patient ID and count no-shows
    no_show_count = (
        noshow.groupby("Patient ID")["Appointment_status"]
        .count()
        .reset_index(name="No_shows")
    )
    df = df.merge(no_show_count, how="left", on="Patient ID").fillna(0)

    print("🔂 Drop Column no longer needed")
    df.drop(
        columns=[
            "Appointment booked date",
            "Appointment date",
            "Booked_by",
            "Clinician",
            "app datetime",
            "Postcode",
            "Registration date",
            "Language",
            "Latitude",
            "Longitude",
            "NHS number",
            "Patient ID",
            "Registration status",
        ],
        inplace=True,
    )

    pre_drop = df.shape[0]
    boolean_mask = df["Rota"] != "DROP"
    # Applying the boolean filteraing
    df = df[boolean_mask].reset_index(drop=True)
    df.reset_index(inplace=True, drop=True)
    post_drop = df.shape[0]
    print(f"🔂 Rows dropped from Rotas other than spec: {pre_drop - post_drop}")

    pre_drop = df.shape[0]
    df.drop(df[df["book_to_app_days"] < 0].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    post_drop = df.shape[0]
    print(f"🔂 Rows from with Negative book_to_app_days: {pre_drop - post_drop}")
    
    print(f"🔂 Drop rows with Sex Unknonw & Indeterminate")
    df = df[~df['Sex'].isin(['Indeterminate', 'Unknown'])]
    
    print(f"🔂 Labelencode Column Sex")
    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

    print(f"🔂 OneHotEncode Column Rota")
    # OneHotEncode Rota
    ohe = OneHotEncoder(handle_unknown="ignore")
    encoded = ohe.fit_transform(df[["Rota"]]).toarray()
    # Create feature names manually
    feature_names = [f"Rota_{category}" for category in ohe.categories_[0]]

    # Convert the encoded array back into a DataFrame
    encoded_data = pd.DataFrame(encoded, columns=feature_names)
    # Concatenate the original DataFrame and the encoded DataFrame
    df = pd.concat([df, encoded_data], axis=1)
    # Drop the original column
    df = df.drop(["Rota"], axis=1)
    print(f"🔂 Extract Ethnicity Category")
    df["Ethnicity category"] = df["Ethnicity category"].fillna("").astype(str)
    df["Ethnicity category"] = df["Ethnicity category"].apply(extract_ethnicity)

    print(f"🔂 OneHotEncode Ethnicity")
    # OneHotEncode Rota
    ohe = OneHotEncoder(handle_unknown="ignore")
    encoded = ohe.fit_transform(df[["Ethnicity category"]]).toarray()
    # Create feature names manually
    feature_names = [f"Ethnicity_{category}" for category in ohe.categories_[0]]
    # Convert the encoded array back into a DataFrame
    encoded_data = pd.DataFrame(encoded, columns=feature_names)
    # Concatenate the original DataFrame and the encoded DataFrame
    df = pd.concat([df, encoded_data], axis=1)
    # Drop the original column
    df = df.drop(["Ethnicity category"], axis=1)

    print("💾 Saving to output_data/full_train_data.csv...")
    df.to_csv(f"{OUTPUT_DATA}full_train_data.csv", index=False)
    end_time = time.time()
    print(f"✅ Done in {round((end_time - start_time),2)} sec {df.shape}")
    return df
