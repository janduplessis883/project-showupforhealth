from showupforhealth.utils import *
from showupforhealth.params import *
from showupforhealth.ml_functions.predict import *
from showupforhealth.interface.model_predict import *


# Your original surgery dictionary
surgery_dict = {
    "ECS": {"model": 6, "threshold": 0.6},
    "HPVM": {"model": 6, "threshold": 0.6},
    "KMC": {"model": 6, 'threshold': 0.6},
    "SMW": {"model": 6, "threshold": 0.6},
    "TGP": {"model": 6, "threshold": 0.6},
    "TCP": {"model": 6, "threshold": 0.6},
}

if __name__ == "__main__":
    if len(sys.argv) > 1:
        week = sys.argv[1]
        webhook = sys.argv[2]
    else:
        webhook = ""
        week = "999"

    # Iterate through the dictionary
    for surgery_prefix, details in surgery_dict.items():
        model = details["model"]
        threshold = details["threshold"]

        print(
            f"{Fore.YELLOW}\n---------------------{surgery_prefix}, Model: {model}, Threshold: {threshold}-----------------------------------------------------------------"
        )
        print(Style.RESET_ALL)

        filepath = f"{UPLOAD_FOLDER}/{surgery_prefix}_predict.csv"
        # Setup Variable
        predictor = SurgeryPredictor()
        predictor.surgery_prefix = surgery_prefix
        predictor.scaler_no = str(model)
        predictor.select_threshold = str(threshold)
        predictor.week = int(week)

        data = predictor.final_predict(surgery_prefix)

        if webhook == "w":
            predictor.send_webhook()
