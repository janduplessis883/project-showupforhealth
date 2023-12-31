from datetime import datetime, timedelta, date
from IPython.display import display, clear_output
import os
import time
from ipywidgets import Dropdown, FileUpload, Button, HBox, Output, VBox, RadioButtons
import pandas as pd
from colorama import Fore, Style, init

init(autoreset=True)

from showupforhealth.interface.model_predict import *
from showupforhealth.params import *
from showupforhealth.ml_functions import *


def generate_weeks(start_date, end_date):
    weeks = []
    # Generate the range of dates for Saturdays within the start_date and end_date
    saturdays = pd.date_range(start=start_date, end=end_date, freq="W-SAT")

    for saturday in saturdays:
        year = saturday.isocalendar()[0]
        week_number = saturday.isocalendar()[1]

        # Pandas date_range includes the last day, so shift back by 6 days to get the Sunday
        sunday = saturday - pd.DateOffset(days=6)

        start_week = sunday.strftime("%d %b")
        end_week = saturday.strftime("%d %b")

        weeks.append(f"{year} Week {week_number} - {start_week} to {end_week}")

    return weeks


# Define date range for the weeks dropdown
start_date = date(2023, 8, 20)
end_date = date(2026, 1, 1)


weeks_of_2023 = generate_weeks(start_date, end_date)
week_dropdown = Dropdown(options=weeks_of_2023, description="Week:")

current_week_number = datetime.now().isocalendar()[1]
current_year = datetime.now().year
current_week = next(
    (
        week
        for week in weeks_of_2023
        if f"{current_year} Week {current_week_number} -" in week
    ),
    None,
)
week_dropdown.value = current_week if current_week else weeks_of_2023[0]


surgery_dict = {
    "Earls Court Medical Centre": "ECMC",
    "Earls Court Surgery": "ECS",
    "Empreros Gate Health Centre": "EGHC",
    "Health Partners Violet Melchett": "HPVM",
    "Kensington Health Centre": "KHC",
    "Knightsbridge Medical Centre": "KMC",
    "Scarsdale Medical Centre": "SMC",
    "Stanhope Mews West": "SMW",
    "The Abingdon Medical Centre": "AMC",
    "The Chelsea Practice": "TCP",
    "The Good Practice": "TGP",
}
surgery_dropdown = Dropdown(options=surgery_dict.keys(), description="Surgery:")
surgery_prefix = None

file_type_buttons = RadioButtons(
    options=["Predict", "Control"], description="File Type:", disabled=False
)


def on_upload_button_click(b):
    selected_week = week_dropdown.value
    surgery_prefix = surgery_dict[surgery_dropdown.value]
    file_type = file_type_buttons.value

    if selected_week and surgery_prefix:
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        for name, file_info in file_upload.value.items():
            new_name = f"{surgery_prefix}_{file_type.lower()}.csv"  # Modified here to add the file_type
            week_folder = os.path.join(
                os.path.expanduser("~"),
                "code",
                "janduplessis883",
                "data-showup",
                "uploads",
                selected_week,
            )
            if not os.path.exists(week_folder):
                os.makedirs(week_folder)
            full_path = os.path.join(week_folder, new_name)
            with open(full_path, "wb") as f:
                f.write(file_info["content"])
            print(f"Uploaded {new_name} with length {len(file_info['content'])} bytes.")
            time.sleep(2)
            clear_output(wait=True)
            display(
                HBox([week_dropdown, surgery_dropdown, file_type_buttons]),
                HBox([file_upload, upload_button]),
            )  # Added the file_type_buttons here


file_upload = FileUpload(description="Upload File")
upload_button = Button(description="Submit Upload")

upload_button.on_click(on_upload_button_click)


# Folder selection Part
def sort_folder(folder):
    parts = folder.split(" ")
    year = int(parts[0])
    week = int(parts[2])
    return (year, week)


# Function to list folders in a given directory
def list_folders_in_directory(directory):
    with os.scandir(directory) as entries:
        folders = [entry.name for entry in entries if entry.is_dir()]
        return sorted(folders, key=sort_folder)


# Function to list files in a selected folder
def list_files_in_folder(folder):
    folder_path = os.path.join(week_folder, folder)
    with os.scandir(folder_path) as entries:
        return [entry.name for entry in entries if entry.is_file()]


# Function to handle dropdown change
current_selected_folder = None


def on_dropdown_change(change):
    global current_selected_folder
    if change["name"] == "value" and (change["new"] is not None):
        current_selected_folder = change["new"]
        with output:
            clear_output()
            print(f"Selected Folder: {current_selected_folder}")
            files = list_files_in_folder(current_selected_folder)
            print("Files in this folder:")
            for f in files:
                print(f)


# Function for Predict button click (to be programmed later)
def on_predict_click(b):
    global current_selected_folder
    if current_selected_folder is not None:
        with output:
            print(f"{Fore.RED}Running predictions on folder {current_selected_folder}")

            folder_path = os.path.join(week_folder, current_selected_folder)
            files = list_files_in_folder(current_selected_folder)

            for file in files:
                file_path = os.path.join(folder_path, file)
                print(f"Running prediction on {file}...")
                data = file_predict(file_path)
                display(data)
                # For instance, read file and perform prediction
    else:
        with output:
            print("No folder selected. Please select a folder.")


# Create dropdown for folder selection
week_folder = os.path.join(
    os.path.expanduser("~"), "code", "janduplessis883", "data-showup", "uploads"
)

folder_names = list_folders_in_directory(week_folder)
folder_dropdown = Dropdown(options=folder_names, description="Select Folder:")
folder_dropdown.observe(on_dropdown_change, names="value")

# Create Predict button
predict_button = Button(description="Predict")
predict_button.on_click(on_predict_click)

# Create Output widget to display information
output = Output()

# Create VBox to hold all widgets and display it
box = VBox([folder_dropdown, predict_button, output])


def file_upload_module():
    return display(
        HBox([week_dropdown, surgery_dropdown, file_type_buttons]),
        HBox([file_upload, upload_button]),
    )


def folder_predict_module():
    return display(box)


if __name__ == "__main__":
    display(HBox([week_dropdown, surgery_dropdown]), HBox([file_upload, upload_button]))
