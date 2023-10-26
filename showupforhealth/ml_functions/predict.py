import requests
import pandas as pd
import time

from showupforhealth.params import *
from showupforhealth.utils import *
from showupforhealth.ml_functions.encoders import *

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def predict_add_weather(surgery_prefix):
    data = pd.read_csv(f"{UPLOAD_FOLDER}/{surgery_prefix}_predict.csv")

    if surgery_prefix == "ECS":
        data["surgery_ECS"] = 1
        data["surgery_HPVM"] = 0
        data["surgery_KMC"] = 0
        data["surgery_SMW"] = 0
        data["surgery_TCP"] = 0
        data["surgery_TGP"] = 0
    elif surgery_prefix == "HPVM":
        data["surgery_ECS"] = 0
        data["surgery_HPVM"] = 1
        data["surgery_KMC"] = 0
        data["surgery_SMW"] = 0
        data["surgery_TCP"] = 0
        data["surgery_TGP"] = 0
    elif surgery_prefix == "KMC":
        data["surgery_ECS"] = 0
        data["surgery_HPVM"] = 0
        data["surgery_KMC"] = 1
        data["surgery_SMW"] = 0
        data["surgery_TCP"] = 0
        data["surgery_TGP"] = 0
    elif surgery_prefix == "SMW":
        data["surgery_ECS"] = 0
        data["surgery_HPVM"] = 0
        data["surgery_KMC"] = 0
        data["surgery_SMW"] = 1
        data["surgery_TCP"] = 0
        data["surgery_TGP"] = 0
    elif surgery_prefix == "TCP":
        data["surgery_ECS"] = 0
        data["surgery_HPVM"] = 0
        data["surgery_KMC"] = 0
        data["surgery_SMW"] = 0
        data["surgery_TCP"] = 1
        data["surgery_TGP"] = 0
    elif surgery_prefix == "TGP":
        data["surgery_ECS"] = 0
        data["surgery_HPVM"] = 0
        data["surgery_KMC"] = 0
        data["surgery_SMW"] = 0
        data["surgery_TCP"] = 0
        data["surgery_TGP"] = 1

    data["weather_time"] = data["Appointment time"].str.split("-").str[0]
    data["weather_datetime"] = data["Appointment date"] + " " + data["weather_time"]
    data["Appointment date"] = pd.to_datetime(data["Appointment date"])
    data["weather_datetime"] = pd.to_datetime(data["weather_datetime"])
    start_date = data["Appointment date"].dt.strftime("%Y-%m-%d").min()
    end_date = data["Appointment date"].dt.strftime("%Y-%m-%d").max()

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

    df = data.merge(api_data, how="left", on="weather_datetime")
    nanindf = df.isna().sum()

    if nanindf.sum() > 0:
        print(f"❌ NaN values in df - ERROR: {nanindf}")
    else:
        df["Patient ID"] = df["Patient ID"].astype("int")
        return df


def predict_add_weather_from_df(df):
    df["weather_time"] = df["Appointment time"].str.split("-").str[0]
    df["Appointment date"] = df["Appointment date"].astype("str")
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

    df = df.merge(api_data, how="left", on="weather_datetime")
    nanindf = df.isna().sum()

    if nanindf.sum() > 0:
        print(f"❌ NaN values in df - ERROR: {nanindf}")
    else:
        df["Patient ID"] = df["Patient ID"].astype("int")
        return df


def predict_add_weather_file(df):
    data = df

    data["weather_time"] = data["Appointment time"].str.split("-").str[0]
    data["weather_datetime"] = data["Appointment date"] + " " + data["weather_time"]
    data["Appointment date"] = pd.to_datetime(data["Appointment date"])
    data["weather_datetime"] = pd.to_datetime(data["weather_datetime"])
    start_date = data["Appointment date"].dt.strftime("%Y-%m-%d").min()
    end_date = data["Appointment date"].dt.strftime("%Y-%m-%d").max()

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

    df = data.merge(api_data, how="left", on="weather_datetime")
    nanindf = df.isna().sum()

    if nanindf.sum() > 0:
        print(f"❌ NaN values in df - ERROR: {nanindf}")
    else:
        df["Patient ID"] = df["Patient ID"].astype("int")
        return df


def predict_add_weather_from_df(df):
    df["weather_time"] = df["Appointment time"].str.split("-").str[0]
    df["Appointment date"] = df["Appointment date"].astype("str")
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

    df = df.merge(api_data, how="left", on="weather_datetime")
    nanindf = df.isna().sum()

    if nanindf.sum() > 0:
        print(f"❌ NaN values in df - ERROR: {nanindf}")
    else:
        df["Patient ID"] = df["Patient ID"].astype("int")
        return df


def predict_add_global_register(df):
    register = pd.read_csv(f"{OUTPUT_DATA}/global_disease_register.csv", dtype="str")
    register["Patient ID"] = register["Patient ID"].astype("int")
    predict_df = df.merge(register, how="left", on="Patient ID")

    incount = predict_df.shape[0]
    predict_df = predict_df.dropna()
    outcount = predict_df.shape[0]

    return predict_df


def predict_feature_engineering(df):
    start_time = time.time()

    df.rename(
        columns={
            "Booked by": "Booked_by",
            "Appointment time": "Appointment_time",
            "Rota type": "Rota",
            "Age in years": "Age",
        },
        inplace=True,
    )

    # Filter rows where 'Registration status' is 'Current'
    df.drop(df[df["Registration status"] == "Deceased, Deducted"].index, inplace=True)
    df.drop(df[df["Registration status"] == "Deceased"].index, inplace=True)
    df.drop(df[df["Registration status"] == "Deducted"].index, inplace=True)

    # Convert Date Columns to DATETIME

    datetime_cols = ["Appointment booked date", "Appointment date", "Registration date"]
    for datetime_col in datetime_cols:
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    df["Appointment_time"] = df["Appointment_time"].astype("str")
    df["Appointment_time"] = df["Appointment_time"].str.split(":").str[0].astype(int)

    df["book_to_app_days"] = (
        df["Appointment date"] - df["Appointment booked date"]
    ).dt.total_seconds() / (60 * 60 * 24)

    # df['Appointment_time'] = df['Appointment_time'].astype('str')
    # df['Appointment_time'] = df['Appointment_time'].str.split(':').str[0].astype(int)

    df["booked_by_clinician"] = (df["Booked_by"] == df["Clinician"]).astype(int)

    df["Rota"] = df["Rota"].map(extract_rota_type)

    df["registered_for_months"] = (
        (pd.Timestamp.now() - df["Registration date"]).dt.total_seconds()
        / (60 * 60 * 24 * 7 * 30)
    ).apply(np.ceil)

    # print('week month day_of_week')
    # df[['week', 'month', 'day_of_week']] = df['Appointment date'].apply(lambda x: pd.Series([x.week, x.month, x.dayofweek]))

    df["week"] = df["Appointment date"].dt.isocalendar().week

    df["month"] = df["Appointment date"].dt.month

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

    df.drop(
        columns=[
            "Appointment booked date",
            "Appointment date",
            "Booked_by",
            "Clinician",
            "Postcode",
            "Registration date",
            "Language",
            "Latitude",
            "Longitude",
            "NHS number",
            "weather_datetime",
            "weather_time",
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

    pre_drop = df.shape[0]
    df.drop(df[df["book_to_app_days"] < 0].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    post_drop = df.shape[0]

    df = df[~df["Sex"].isin(["Indeterminate", "Unknown"])]

    le = LabelEncoder()
    df["Sex"] = le.fit_transform(df["Sex"])

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

    df["Ethnicity category"] = df["Ethnicity category"].fillna("").astype(str)
    df["Ethnicity category"] = df["Ethnicity category"].apply(extract_ethnicity)

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

    df.dropna(inplace=True)

    end_time = time.time()

    return df


def add_noshows(df):
    noshows = pd.read_csv(f"{OUTPUT_DATA}/no_shows_db.csv")
    merged = df.merge(noshows, on="Patient ID", how="left")
    countin = merged.shape[0]
    merged = merged.fillna(0)
    countout = merged.shape[0]

    return merged


def test_predict(df):
    no_columns = df.shape[1]
    if no_columns != 37:
        print(
            "⛔️ TEST FAILED - df not 37 columns, inspect! ? Appointment_status column1"
        )
        # display(df.head())


def get_appointment_info(surgery_prefix):
    df = pd.read_csv(f"{UPLOAD_FOLDER}/{surgery_prefix}_predict.csv")
    pt_id = df[["Appointment Date", "Appointment time", "Patient ID"]]
    return pt_id


def make_predict():
    surgery_prefix = input("Enter Surgery Prefix: ")
    df = predict_add_weather(surgery_prefix=surgery_prefix)
    df = predict_add_global_register(df)
    df = add_noshows(df)
    df = predict_feature_engineering(df)
    test_predict(df)
    return df


def streamlit_predict(surgery_prefix):
    print("✅ streamlit_predict add weather")
    df = predict_add_weather(surgery_prefix=surgery_prefix)
    print("✅ streamlit_predict add global register")
    df = predict_add_global_register(df)
    print("✅ streamlit_predict add no shows")
    df = add_noshows(df)
    print("✅ streamlit_predict feature Engineering")
    df = predict_feature_engineering(df)
    test_predict(df)
    print("✅ streamlit_predict output DF")
    return df


def streamlit_predict2(df):
    df = predict_add_weather_from_df(df)
    df = predict_add_global_register(df)
    df = add_noshows(df)
    df = predict_feature_engineering(df)
    test_predict(df)
    return df


if __name__ == "__main__":
    make_predict()
