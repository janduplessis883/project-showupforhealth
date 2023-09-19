import pandas as pd
import time

from showupforhealth.params import *
from showupforhealth.ml_functions.encoders import haversine_distance


def make_disease_register(surgery_list=["ECS", "TCP", "TGP", "SMW", "KMC", "HPVM"]):
    start_time = time.time()
    print(
        "==== Preparing Global Disease Register + IMD2023 info =========================="
    )

    disease_register = []
    for surgery in surgery_list:
        register_path = f"{RAW_DATA}/{surgery}/{surgery}"

        idnhs = pd.read_excel(f"{register_path}_NHS_PTID.xlsx", dtype="str")
        idnhs.dropna(inplace=True)
        frail = pd.read_csv(f"{register_path}_FRAILTY.csv", dtype="str")
        dep = pd.read_csv(f"{register_path}_DEPRESSION.csv", dtype="str")
        obesity = pd.read_csv(f"{register_path}_OBESITY.csv", dtype="str")
        chd = pd.read_csv(f"{register_path}_IHD.csv", dtype="str")
        dm = pd.read_csv(f"{register_path}_DM.csv", dtype="str")
        hpt = pd.read_csv(f"{register_path}_HPT.csv", dtype="str")
        ndhg = pd.read_csv(f"{register_path}_NDHG.csv", dtype="str")
        smi = pd.read_csv(f"{register_path}_SMI.csv", dtype="str")

        ptid = idnhs.merge(frail, how="left", on="NHS number")
        # ptid = ptid.drop(columns='NHS number')

        register = (
            ptid.merge(dep, how="left", on="Patient ID")
            .merge(obesity, how="left", on="Patient ID")
            .merge(chd, how="left", on="Patient ID")
            .merge(dm, how="left", on="Patient ID")
            .merge(hpt, how="left", on="Patient ID")
            .merge(ndhg, how="left", on="Patient ID")
            .merge(smi, how="left", on="Patient ID")
            .fillna(0)
        )
        print(f"💊 {surgery} Disease Register completed", end=" ")
        # Add IMD and distance from station
        imd = pd.read_csv(IMD_DATA)

        full_register = register.merge(imd, how="left", on="Postcode")
        print(f" - {surgery} IMD2023 added")
        full_register["distance_to_surg"] = full_register.apply(
            lambda row: haversine_distance(surgery, row["Latitude"], row["Longitude"]),
            axis=1,
        )
        disease_register.append(full_register)

    global_register = pd.concat(disease_register, axis=0, ignore_index=True)
    print(f"🔂 Concat Registers")
    print(f"❌ Drop NaN")
    global_register.dropna(inplace=True)

    print("❌ Drop Duplicates on column Patient ID")
    global_register.drop_duplicates(subset="Patient ID", keep="last", inplace=True)

    print("💾 Saving to output_data/global_disease_register.csv...")
    register_out = f"{OUTPUT_DATA}/global_disease_register.csv"
    global_register.to_csv(register_out, index=False)

    end_time = time.time()
    print(f"✅ Done in {round((end_time - start_time),2)} sec {global_register.shape}")
    return global_register


if __name__ == "__main__":
    make_disease_register()
