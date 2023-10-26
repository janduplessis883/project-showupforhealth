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
import requests
import json
import sys
import os
import base64
import warnings

warnings.filterwarnings("ignore")


# Now, the surgery_prefix variable contains the value passed from the command line


class SurgeryPredictor:
    def __init__(self):
        self.week = 0
        self.surgery_prefix = ""
        self.surgery = ""
        self.scaler_no = ""
        self.select_threshold = ""
        self.unique_dna_no = 0
        self.heatmap = ""
        self.duplicates = 0

    # Define your custom serialization function
    def json_serial(self, obj):
        if isinstance(obj, np.int64):
            return int(obj)
        raise TypeError(f"Type {type(obj)} not serializable")

    def send_webhook(self):
        webhook_url = "https://eo6sfmvkbnp22n7.m.pipedream.net"
        data = {
            "week": int(self.week),
            "surgery": self.surgery,
            "surgery_prefix": self.surgery_prefix,
            "model": self.scaler_no,
            "threshold": self.select_threshold,
            "predicted_count": int(self.predicted_count),
            "unique_dna_no": int(self.unique_dna_no),
            "duplicates": int(self.duplicates),
            "heatmap": self.heatmap,
        }
        response = requests.post(
            webhook_url,
            data=json.dumps(data, default=self.json_serial),
            headers={"Content-Type": "application/json"},
        )
        if response.status_code != 200:
            raise ValueError(
                f"Request to webhook failed: {response.status_code}, {response.text}"
            )
        else:
            print(f"🛜 Successfully sent webhook")

    def set_surgery(self):
        if self.surgery_prefix == "ECS":
            self.surgery = "Earls Court Surgery"
        elif self.surgery_prefix == "TGP":
            self.surgery = "The Good Practice"
        elif self.surgery_prefix == "SMW":
            self.surgery = "Stanhope Mews West"
        elif self.surgery_prefix == "TCP":
            self.surgery = "The Chelsea Practice"
        elif self.surgery_prefix == "KMC":
            self.surgery = "Knightsbridge Medical Centre"
        elif self.surgery_prefix == "HPVM":
            self.surgery = "Health Partners Violet Melchett"
        else:
            self.surgery = ""

    def f1_score(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        recall = true_positives / (possible_positives + K.epsilon())
        f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
        return f1_val

    def sort_df_columns(self, df):
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
                "surgery_ECS",
                "surgery_HPVM",
                "surgery_KMC",
                "surgery_SMW",
                "surgery_TCP",
                "surgery_TGP",
            ]
        ]
        return new_df

    def clinic_heatmap(self, data):
        # Assuming your DataFrame is called 'data'
        # Convert 'Appointment date' to datetime format
        plt.figure(figsize=(16, 8))
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
        ax = sns.heatmap(
            heatmap_data, cmap="Blues", annot=True
        )  # annot=True will display the counts on each cell of the heatmap

        # Set the aspect ratio
        ax.set_aspect("auto")

        # Save the plot to the local UPLOAD_FOLDER directory
        plot_filename = f"{UPLOAD_FOLDER}/{self.surgery_prefix}_clinic_HEATMAP.png"
        plt.savefig(
            plot_filename, bbox_inches="tight"
        )  # bbox_inches='tight' helps to fit the plot
        # Show the plot (optional, if you still want to display it after saving)
        # plt.show()
        print(f"💾🔥 Heatmap saved: {plot_filename}")
        return plot_filename

    def scaler_model_predict(self, df):
        print(f"Debug - scaler_no: {self.scaler_no} of Type: {type(self.scaler_no)}")
        print(Fore.GREEN + "\n👉 Select Scaler + Model:")
        print(Fore.GREEN + "1. ❌Jan-Model-128-256-16-32-16Sept23")
        print(Fore.GREEN + "2. ❌Jan-Backup-128-256-16-32-22Sept23")
        print(Fore.GREEN + "3. ❌Experiment-256-512-16-32-undersample-0.15")
        print(Fore.GREEN + "4. model_youthfull_darkness_9oct23")
        print(Fore.GREEN + "5. model_9oct23_3")
        print(Fore.GREEN + "6. DeepNetwork_15oct23vlr")
        print(Fore.GREEN + "7. Wobbly_21oct23vlr")

        if self.scaler_no == "":
            self.scaler_no = input(Fore.RED + "Enter Selection: ")
        print(f"Model + Scaler No Selected: {self.scaler_no}")
        print(Style.RESET_ALL)

        scaler_path = {
            "1": "jan_scaler_17sept23.pkl",
            "2": "jan_scaler_22sept239am1664.pkl",
            "3": "jan_scaler_22sept239am1664.pkl",
            "4": "jan_scaler_9oct23.pkl",
            "5": "jan_scaler_9oct23_3.pkl",
            "6": "jan_scaler_6_15oct23.pkl",
            "7": "jan_scaler_6_15oct23.pkl",
        }

        model_path = {
            "1": "model16sept23_jan.h5",
            "2": "model_weights_9am16642023-09-22 09-01-10.h5",
            "3": "model_weights_gentle_water2023-09-22 10-37-13.h5",
            "4": "model_youthfull_darkness_9oct23.h5",
            "5": "model_9oct23_3.h5",
            "6": "model_6_128b_vlr_15oct23.h5",
            "7": "model_7_wobbly_21oct23.h5",
        }

        if self.scaler_no in scaler_path:
            print("Debug - Inside the if condition")
            scaler = load(f"{MODEL_OUTPUT}/{scaler_path[self.scaler_no]}")
            model = load_model(
                f"{MODEL_OUTPUT}/{model_path[self.scaler_no]}",
                custom_objects={"f1_score": f1_score},
            )

            df = self.sort_df_columns(df)
            df = transform_data(df, scaler)
            df = df.astype("float32")
            predictions = model.predict(df)
            return predictions

        else:
            print("Not a valid scaler + model selection.")
            print("Debug - Not a valid scaler + model selection")
            return None

    def display_threshold_info(self, predictions, thresholds=[0.5, 0.6, 0.7, 0.8]):
        print()
        for t in thresholds:
            class_labels = (predictions > t).astype(int)
            check_mark = "---✅" if str(t) == self.select_threshold else ""

            print(
                f"{Fore.BLUE}DNAs predicted at {t} threshold: {class_labels.flatten().tolist().count(0)}{check_mark}"
            )
        if self.select_threshold == "":
            self.select_threshold = input(Fore.RED + "Select a Threshold: ")

        print(Style.RESET_ALL)
        class_labels = (predictions > float(self.select_threshold)).astype(int)
        self.predicted_count = int(class_labels.flatten().tolist().count(0))
        return class_labels

    def display_outcome_df(self, class_labels, pt_id_df):
        class_labels_list = class_labels.flatten().tolist()
        new_col = pd.Series(class_labels_list, name="Model_Prediction")

        df = pd.concat([pt_id_df, new_col], axis=1)
        no_shows = df[df["Model_Prediction"] == 0]
        self.duplicates = no_shows.duplicated().sum()
        unique_no_snows = no_shows.drop_duplicates(subset="Patient ID")
        # print("-- Unique Predicted No Shows --------------------------")
        print(f"⛔️ Duplicates Dropped: {self.duplicates}")
        return unique_no_snows

    def make_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            base64_image = base64.b64encode(image_file.read()).decode()

        return base64_image

    def final_predict(self, new_surgery_prefix):
        self.surgery_prefix = new_surgery_prefix
        self.set_surgery()
        print("✅ Starting streamlit_predict...")
        X_temp = streamlit_predict(self.surgery_prefix)
        X_new = X_temp.drop(columns="Patient ID")
        pt_id = X_temp[["Patient ID"]]

        predictions = self.scaler_model_predict(X_new)

        class_labels = self.display_threshold_info(predictions)
        print("✅ Finished Pre-processing")
        df = self.display_outcome_df(class_labels, pt_id)
        print("\n👉 Original Class Labels - un-# to display values here")
        # display(df)

        surgery = pd.read_csv(f"{UPLOAD_FOLDER}/{self.surgery_prefix}_predict.csv")

        new = surgery.merge(df, on="Patient ID", how="left")
        new["Model_Prediction"] = new["Model_Prediction"].replace(0.0, "Predicted DNA")
        new.dropna(subset="Model_Prediction", inplace=True)

        new.drop(columns=["Appointment booked date", "Booked by"], inplace=True)

        new.sort_values(by=["Appointment date", "Appointment time"], inplace=True)

        # Merge with SMS List to get patient details
        sms_allpts = pd.read_csv(
            f"{SMS_FOLDER}/{self.surgery_prefix}_smslist.csv", encoding="latin"
        )
        prediction_pts = new.merge(sms_allpts, on="Patient ID", how="left")
        file_path = f"{UPLOAD_FOLDER}/{self.surgery_prefix}_prediction_OUTPUT.csv"
        prediction_pts.to_csv(file_path, index=False)
        print(f"💾🔮 Prediction Output saved ❌ Further work required: {UPLOAD_FOLDER}")

        accurx_list = prediction_pts[
            ["NHS number", "Preferred telephone number", "Date of birth", "First name"]
        ]
        accurx_list.drop_duplicates(inplace=True)
        accurx_path = f"{UPLOAD_FOLDER}/{self.surgery_prefix}_ACCURX_bulk_sms_list.csv"
        accurx_list.to_csv(accurx_path, index=False)
        print(f"💾📞 Accurx Bulk SMS List saved: {accurx_path}")

        if "Appointment status" in new.columns:
            print(new[["Appointment status"]].value_counts())
        try:
            unique_dna_no = new[["Appointment status"]].value_counts()['Did Not Attend']
            self.unique_dna_no = int(unique_dna_no)
        except (IndexError, KeyError):
            unique_dna_no = 0  # or some other default value

        try:
            heatmap_path = self.clinic_heatmap(prediction_pts)
            self.heatmap = self.make_base64(heatmap_path)
        except (ValueError, IndexError, KeyError):
            pass
        
        return prediction_pts

    def make_accurx_list(self, df):
        sms_allpts = pd.read_csv(
            f"{SMS_FOLDER}/{self.surgery_prefix}_smslist.csv", encoding="latin"
        )
        sms = df.merge(sms_allpts, on="Patient ID", how="left")
        accurx_list = sms[
            ["NHS number", "Preferred telephone number", "Date of birth", "First name"]
        ]

        return accurx_list


if __name__ == "__main__":
    if len(sys.argv) > 1:
        surgery_prefix = sys.argv[1]
        week = sys.argv[2]
    else:
        surgery_prefix = "SMW"
        week = 999

    filepath = f"{UPLOAD_FOLDER}/{surgery_prefix}_predict.csv"
    assert os.path.exists(filepath), f"The file {filepath} does not exist."

    predictor = SurgeryPredictor()
    predictor.week = int(week)
    data = predictor.final_predict(surgery_prefix)
    predictor.send_webhook()
