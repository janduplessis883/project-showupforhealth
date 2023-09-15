import pandas as pd
import time

from showupforhealth.params import *
from showupforhealth.ml_functions.disease_register import make_disease_register
from showupforhealth.ml_functions.preprocessor import *


def create_global_appointments_list(
    surgery_list=["ECS", "TCP", "TGP", "SMW", "KMC", "HPVM"]
):
    print(
        "\n==== Processing Appointments data + Add 🌤️ + merge disease_register ==========="
    )
    full_app_list = []
    for surgery_prefix in surgery_list:
        print(f"⏺️ {surgery_prefix} -", end=" ")
        df_list = []
        for i in range(1, 10, 1):
            app = pd.read_csv(f"{RAW_DATA}/{surgery_prefix}/{surgery_prefix}_APP{i}.csv")
            print(f"df {i} ", end=" ")
            df_list.append(app)

        appointments = pd.concat(df_list, axis=0, ignore_index=True)
        print(f"duplicates: {appointments.duplicated().sum()}")
        full_app_list.append(appointments)

        global_appointments = pd.concat(full_app_list, axis=0, ignore_index=True)

    print(f"🔂 Concat Appointments {global_appointments.shape}")
    # Filter and drop rows with 'DROP' value
    return global_appointments


def add_weather(global_apps):
    weather = pd.read_csv(WEATHER_DATA)

    weather["app datetime"] = pd.to_datetime(weather["app datetime"])

    global_apps["app datetime"] = pd.to_datetime(
        global_apps["Appointment date"]
        + " "
        + global_apps["Appointment time"].str.split(expand=True)[0]
    )

    global_apps["app datetime"] = pd.to_datetime(global_apps["app datetime"])

    global_apps_weather = global_apps.merge(weather, how="left", on="app datetime")
    print(f"🌤️ Weather Added to Apps {global_apps_weather.shape}")
    return global_apps_weather


def make_full_preprocess_data():
    start_time = time.time()
    register = make_disease_register()
    register["Patient ID"] = register["Patient ID"].astype("int64")
    apps = create_global_appointments_list()

    apps_weather = add_weather(apps)
    apps_weather["Patient ID"] = apps_weather["Patient ID"].astype("int64")
    full_df = apps_weather.merge(register, how="left", on="Patient ID")
    print(f"↔️ Merge Appointments and Global_Disease_Register {full_df.shape}")
    full_df.dropna(inplace=True)
    print(f"❌ Drop NaN {full_df.shape}")
    full_path = f"{OUTPUT_DATA}/full_preprocess_data.csv"

    print(f"💾 Saving to output_data/full_preprocess_data.csv...")
    full_df.to_csv(full_path, index=False)
    end_time = time.time()
    print(f"✅ Done in {round((end_time - start_time),2)} sec {full_df.shape}")
    return full_df




if __name__ == "__main__":
    data = make_full_preprocess_data()
    feature_engineering(data)
