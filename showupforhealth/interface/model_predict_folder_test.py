from showupforhealth.utils import *
from showupforhealth.params import *
from showupforhealth.ml_functions.predict import *
from showupforhealth.interface.model_predict import *

# ❌❌❌❌❌❌❌❌❌❌❌❌❌❌ Set week and webhook preference
week = 40
webhook = "w"
# ❌❌❌❌❌❌❌❌❌❌❌❌❌❌ Set week and webhook preference
# Define the lists
surgery_prefixes = ["TGP"]
test_models = [7]  # Add more model numbers as you want
test_thresholds = [0.6, 0.7, 0.8]  # Add more thresholds as you want


if __name__ == "__main__":
    # Nested loops to iterate through every combination
    for surgery_prefix in surgery_prefixes:
        for test_model in test_models:
            for test_threshold in test_thresholds:
                print(
                    f"{Fore.YELLOW}\n--------------------- {surgery_prefix}, Model: {test_model}, Threshold: {test_threshold} -----------------------------------------------------------------"
                )
                print(Style.RESET_ALL)

                filepath = f"{UPLOAD_FOLDER}/{surgery_prefix}_predict.csv"
                predictor = SurgeryPredictor()

                # Setting the properties based on the current test run
                predictor.surgery_prefix = surgery_prefix
                predictor.scaler_no = str(test_model)
                predictor.select_threshold = str(test_threshold)
                predictor.week = int(week)  # Assuming week is defined somewhere

                data = predictor.final_predict(surgery_prefix)

                if webhook == "w":  # Assuming webhook is defined somewhere
                    predictor.send_webhook()
