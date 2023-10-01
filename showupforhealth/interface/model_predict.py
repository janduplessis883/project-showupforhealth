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



# Now, the surgery_prefix variable contains the value passed from the command line


class SurgeryPredictor:
    def __init__(self):
        self.surgery_prefix = ''
        self.surgery = ''
        self.scaler_no = ''
        self.select_threshold = ''
        self.unique_dna_no = 0

    def send_webhook(self):
        webhook_url = "https://eo6sfmvkbnp22n7.m.pipedream.net"
        data = {
            "surgery": self.surgery,
            "surgery_prefix": self.surgery_prefix,
            "model": self.scaler_no,
            "threshold": self.select_threshold,
            "predicted_count": self.predicted_count
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

    def set_surgery(self):
        if self.surgery_prefix == 'ECS':
            self.surgery = 'Earls Court Surgery'
        elif self.surgery_prefix == 'TGP':
            self.surgery = 'The Good Practice'
        elif self.surgery_prefix == 'SMW':
            self.surgery = 'Stanhope Mews West'
        else:
            self.surgery = ''

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
        plot_filename = f"{UPLOAD_FOLDER}/{self.surgery_prefix}_clinic_heatmap.png"
        plt.savefig(plot_filename, bbox_inches='tight')  # bbox_inches='tight' helps to fit the plot

        # Show the plot (optional, if you still want to display it after saving)
        plt.show()

        print(f"The heatmap has been saved as {plot_filename}")



    def scaler_model_predict(self, df):
        print(Fore.GREEN + "\n*️⃣ Select Scaler + Model:")
        print(Fore.GREEN + "1. Jan-Model-128-256-16-32-16Sept23")
        print(Fore.GREEN + "2. Jan-Backup-128-256-16-32-22Sept23")
        print(Fore.GREEN + "3. Experiment-256-512-16-32-undersample-0.15")
        
        self.scaler_no = input(Fore.RED + "Enter Selection: ")

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

        if self.scaler_no in scaler_path:
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
            return None

    def display_threshold_info(self, predictions, thresholds=[0.5, 0.6, 0.7, 0.8]):
        print()
        for t in thresholds:
            class_labels = (predictions > t).astype(int)
            
            print(
                f"{Fore.BLUE}DNAs predicted at {t} threshold: {class_labels.flatten().tolist().count(0)}"
            )

        self.select_threshold = input(Fore.RED + "Select a Threshold: ")
        print(Style.RESET_ALL)
        class_labels = (predictions > float(self.select_threshold)).astype(int)
        self.predicted_count = class_labels.flatten().tolist().count(0)
        return class_labels

    def display_outcome_df(self, class_labels, pt_id_df):
        class_labels_list = class_labels.flatten().tolist()
        new_col = pd.Series(class_labels_list, name="Model_Prediction")

        df = pd.concat([pt_id_df, new_col], axis=1)
        no_shows = df[df["Model_Prediction"] == 0]
        count_dup = no_shows.duplicated().sum()
        unique_no_snows = no_shows.drop_duplicates(subset="Patient ID")
        # print("-- Unique Predicted No Shows --------------------------")
        print(f"⛔️ Duplicates Dropped: {count_dup}")
        return unique_no_snows

    def final_predict(self, new_surgery_prefix):
        self.surgery_prefix = new_surgery_prefix
        self.set_surgery()
        print("✅ Starting streamlit_predict...")
        X_temp = streamlit_predict(self.surgery_prefix)
        X_new = X_temp.drop(columns="Patient ID")
        pt_id = X_temp[["Patient ID"]]
        print("✅ scaler_model_predict")
        predictions = self.scaler_model_predict(X_new)
        print("✅ Display Threshold Info")
        class_labels = self.display_threshold_info(predictions)
        print("✅ display Outcome DF")
        df = self.display_outcome_df(class_labels, pt_id)
        print("*️⃣ Original Class Labels")

        
        surgery = pd.read_csv(f"{UPLOAD_FOLDER}/{self.surgery_prefix}_predict40.csv")

        new = surgery.merge(df, on="Patient ID", how="left")
        new["Model_Prediction"] = new["Model_Prediction"].replace(0.0, "Predicted DNA")
        new.dropna(subset="Model_Prediction", inplace=True)

        new.drop(columns=["Appointment booked date", "Booked by"], inplace=True)

        new.sort_values(by=["Appointment date", "Appointment time"], inplace=True)

        new.to_csv(f"{UPLOAD_FOLDER}/{self.surgery_prefix}_prediction_output.csv", index=False)
        
        self.send_webhook()
        return new




if __name__ == "__main__":
    predictor = SurgeryPredictor()
    
    if len(sys.argv) > 1:
        surgery_prefix = sys.argv[1]
    else:
        surgery_prefix = "ECS"
        
    data = predictor.final_predict(surgery_prefix)
    predictor.clinic_heatmap(data)


