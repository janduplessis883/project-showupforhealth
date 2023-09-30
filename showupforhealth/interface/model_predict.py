import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from joblib import dump, load
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K
from keras.models import load_model
from colorama import Fore, Style

from showupforhealth.utils import *
from showupforhealth.params import *
from showupforhealth.ml_functions.predict import *
import time
import requests
import json

# define global variables
scaler_no = 0
select_threshold = 0.0
global surgery_prefix
surgery_prefix = ''
surgery = ''
unique_dna_no = 0
if surgery_prefix == 'ECS':
    surgery == 'Earls Court Surgery'
elif surgery_prefix == 'TGP':
    surgery == 'The Good Practice'
elif surgery_prefix == 'SMW':
    surgery == 'Stanhope Mews West'
else:
    surgery == ''

def send_webhook(surgery, surgery_prefix, model, threshold, predicted_count):
    webhook_url = "https://eo6sfmvkbnp22n7.m.pipedream.net"
    
    data = {
        "surgery": surgery,
        "surgery_prefix": surgery_prefix,
        "model": model,
        "threshold": threshold,
        "predicted_count": predicted_count
    }
    
    response = requests.post(
        webhook_url, data=json.dumps(data),
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        raise ValueError(
            f"Request to webhook failed: {response.status_code}, {response.text}"
        )
    else:
        print(f"Successfully sent webhook")


def time_it(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} took {end_time - start_time:.6f} seconds to run.")
        return result

    return wrapper


def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


def sort_df_columns(df):
    new_df = df[
        [
            "temp",
            "precipitation",
            "Age",
            "Sex",
            "FRAILTY",
            "DEPRESSION",
            "OBESITY",
            "IHD",
            "DM",
            "HPT",
            "NDHG",
            "SMI",
            "IMD2023",
            "dist_to_station",
            "distance_to_surg",
            "book_to_app_days",
            "booked_by_clinician",
            "registered_for_months",
            "sin_week",
            "cos_week",
            "sin_Appointment_time",
            "cos_Appointment_time",
            "sin_month",
            "cos_month",
            "sin_day_of_week",
            "cos_day_of_week",
            "No_shows",
            "Rota_ARRS",
            "Rota_GP",
            "Rota_HCA",
            "Rota_Nurse",
            "Ethnicity_Asian",
            "Ethnicity_Black",
            "Ethnicity_Mixed",
            "Ethnicity_Other",
            "Ethnicity_White",
        ]
    ]
    return new_df


def scaler_model_predict(df):
    print(Fore.GREEN + "\n▶️ Select Scaler + Model:")
    print(Fore.GREEN + "1. jan_scaler_17sept23 + model16sept23_jan")
    print(Fore.GREEN + "2. Jan Backup 22 Sept 9am")
    print(Fore.GREEN + "3. Gentle Water undersample 0.15")
    
    scaler_no = input(Fore.RED + "Enter Selection: ")

    print(Style.RESET_ALL)

    scaler_path = {
        "1": "jan_scaler_17sept23.pkl",
        "2": "jan_scaler_22sept239am1664.pkl",
        "3": "jan_scaler_22sept239am1664.pkl"
    }
    
    model_path = {
        "1": "model16sept23_jan.h5",
        "2": "model_weights_9am16642023-09-22 09-01-10.h5",
        "3": "model_weights_gentle_water2023-09-22 10-37-13.h5"
    }

    if scaler_no in scaler_path:
        scaler = load(f"{MODEL_OUTPUT}/{scaler_path[scaler_no]}")
        model = load_model(
            f"{MODEL_OUTPUT}/{model_path[scaler_no]}",
            custom_objects={"f1_score": f1_score},
        )

        df = sort_df_columns(df)
        df = transform_data(df, scaler)
        df = df.astype("float32")
        predictions = model.predict(df)
        return predictions

    else:
        print("Not a valid scaler + model selection.")
        return None



def display_threshold_info(predictions, thresholds=[0.5, 0.6, 0.7, 0.8]):
    for t in thresholds:
        class_labels = (predictions > t).astype(int)
        print(
            f"No shows Predicted at {t} threshold: {class_labels.flatten().tolist().count(0)}"
        )
    global select_threshold
    select_threshold = input("Select a Threshold: ")
    class_labels = (predictions > float(select_threshold)).astype(int)
    send_webhook("The Good Practice", "surgery_prefix", scaler_no, select_threshold, unique_dna_no)
    return class_labels


def display_outcome_df(class_labels, pt_id_df):
    class_labels_list = class_labels.flatten().tolist()
    new_col = pd.Series(class_labels_list, name="Model_Prediction")

    df = pd.concat([pt_id_df, new_col], axis=1)
    no_shows = df[df["Model_Prediction"] == 0]
    count_dup = no_shows.duplicated().sum()
    unique_no_snows = no_shows.drop_duplicates(subset="Patient ID")
    # print("-- Unique Predicted No Shows --------------------------")
    # print(f"Duplicates Dropped: {count_dup}")
    global unique_dna_no
    unique_dna_no = unique_no_snows.shape[0]
    return unique_no_snows


def clinic_heatmap(data):
    # Assuming your DataFrame is called 'data'
    # Convert 'Appointment date' to datetime format
    plt.figure(figsize=(12, 6))
    data["Appointment date"] = pd.to_datetime(data["Appointment date"])

    # Keep only the date portion (this will remove time, if any)
    data["Appointment date"] = data["Appointment date"].dt.date

    # Create the heatmap data by grouping by 'Clinician' and 'Appointment date'
    heatmap_data = (
        data[data["Model_Prediction"] == "Predicted DNA"]
        .groupby(["Clinician", "Appointment date"])
        .size()
        .reset_index(name="Counts")
    )

    # Pivot the DataFrame so that 'Clinician' is on the Y-axis and 'Appointment date' is on the X-axis
    heatmap_data = heatmap_data.pivot("Clinician", "Appointment date", "Counts")

    # Plot the heatmap
    sns.heatmap(
        heatmap_data, cmap="Blues", annot=True
    )  # annot=True will display the counts on each cell of the heatmap
    plt.show()


def final_predict(surgery_prefix):
    
    print("✅ Starting streamlit_predict...")
    X_temp = streamlit_predict(surgery_prefix)
    X_new = X_temp.drop(columns="Patient ID")
    pt_id = X_temp[["Patient ID"]]
    print("✅ scaler_model_predict")
    predictions = scaler_model_predict(X_new)
    print("✅ Display Threshold Info")
    class_labels = display_threshold_info(predictions)
    print("✅ display Outcome DF")
    df = display_outcome_df(class_labels, pt_id)
    print("❌ Original Class Labels")
    print(df)
    
    surgery = pd.read_csv(f"{UPLOAD_FOLDER}/{surgery_prefix}_predict40.csv")

    new = surgery.merge(df, on="Patient ID", how="left")
    new["Model_Prediction"] = new["Model_Prediction"].replace(0.0, "Predicted DNA")
    new.dropna(subset="Model_Prediction", inplace=True)

    new.drop(columns=["Appointment booked date", "Booked by"], inplace=True)

    new.sort_values(by=["Appointment date", "Appointment time"], inplace=True)

    new.to_csv(f"{UPLOAD_FOLDER}/{surgery_prefix}_prediction_output.csv", index=False)
    return new


def file_predict(file_path):
    df = pd.read_csv(file_path)
    X_temp = streamlit_predict2(df)
    X_new = X_temp.drop(columns="Patient ID")
    pt_id = X_temp[["Patient ID"]]
    predictions = scaler_model_predict(X_new)
    class_labels = display_threshold_info(predictions)
    df = display_outcome_df(class_labels, pt_id)

    surgery = pd.read_csv(file_path)
    new = surgery.merge(df, on="Patient ID", how="left")
    new["Model_Prediction"] = new["Model_Prediction"].replace(0.0, "Predicted DNA")
    new.dropna(inplace=True)
    new.to_csv(file_path + ".PREDICT.csv")
    return new


if __name__ == "__main__":
    print(Fore.BLUE + "\nECS - Earls Court Surgery")
    print(Fore.BLUE + "TGP - The Good Practice")
    print(Fore.BLUE + "TCP - The Chelsea Practice")
    print(Fore.BLUE + "SMW - Stanhope Mews West")
    print(Fore.BLUE + "KMC - Knightsbridge Medical Centre")
    print(Fore.BLUE + "HPVN - Health Partners at Violet Melchett")
    surgery_prefix = input(Fore.RED + "Enter Surgery Prefix to continue: ")
    print(Style.RESET_ALL)

    print("✅ Starting streamlit_predict...")
    X_temp = streamlit_predict(surgery_prefix)
    X_new = X_temp.drop(columns="Patient ID")
    pt_id = X_temp[["Patient ID"]]
    print("✅ scaler_model_predict")
    predictions = scaler_model_predict(X_new)
    print("✅ Display Threshold Info")
    class_labels = display_threshold_info(predictions)
    print("✅ display Outcome DF")
    df = display_outcome_df(class_labels, pt_id)
    
    print("❌ Original Class Labels")
    print(df)
    surgery = pd.read_csv(f"{UPLOAD_FOLDER}/{surgery_prefix}_predict40.csv")

    new = surgery.merge(df, on="Patient ID", how="left")
    new["Model_Prediction"] = new["Model_Prediction"].replace(0.0, "Predicted DNA")
    new.dropna(subset="Model_Prediction", inplace=True)

    new.drop(columns=["Appointment booked date", "Booked by"], inplace=True)

    new.sort_values(by=["Appointment date", "Appointment time"], inplace=True)

    new.to_csv(f"{UPLOAD_FOLDER}/{surgery_prefix}_prediction_output.csv", index=False)



