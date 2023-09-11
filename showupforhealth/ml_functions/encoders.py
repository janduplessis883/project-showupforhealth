import math
import re


def haversine_distance(surgery_prefix, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers

    if surgery_prefix == "ECS":
        lat1, lon1 = 51.488721, -0.191873
    elif surgery_prefix == "SMW":
        lat1, lon1 = 51.494474, -0.181931
    elif surgery_prefix == "TCP":
        lat1, lon1 = 51.48459, -0.171887
    elif surgery_prefix == "HPVM":
        lat1, lon1 = 51.48459, -0.171887
    elif surgery_prefix == "KMC":
        lat1, lon1 = 51.49807, -0.159918
    elif surgery_prefix == "TGP":
        lat1, lon1 = 51.482652, -0.178066

    # Convert degrees to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Differences
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    # Haversine formula
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance  # in kilometers


def extract_rota_type(text):
    # HOW TO APPLY IT
    # Apply extract_role function and overwrite Rota type column
    # full_appointments['Rota type'] = full_appointments['Rota type'].apply(extract_rota_type)
    role_map = {
        "GP": [
            "GP",
            "Registrar",
            "Urgent",
            "Telephone",
            "111",
            "FY2",
            "F2",
            "Extended Hours",
            "GP Clinic",
            "Session",
        ],
        "Nurse": ["Nurse", "Nurse Practitioner"],
        "HCA": ["HCA", "Health Care Assistant", "Phlebotomy"],
        "ARRS": [
            "Pharmacist",
            "Paramedic",
            "Physiotherapist",
            "Physicians Associate",
            "ARRS",
            "PCN",
        ],
    }

    for role, patterns in role_map.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return role
    return "DROP"


def fix_appointment_status(status):
    """
    Function to categorize appointment statuses into binary format.

    Args:
        status (str): Appointment status.

    Returns:
        int: Returns 1 if status is in ['In Progress', 'Arrived', 'Patient Walked Out', 'Finished', 'Waiting'], 0 if status is 'Did Not Attend' or 'ERROR' otherwise.
    """
    if status in [
        "In Progress",
        "Arrived",
        "Patient Walked Out",
        "Finished",
        "Waiting",
    ]:
        return 1
    elif status == "Did Not Attend":
        return 0


def extract_ethnicity(text):
    # HOW TO APPLY IT
    # Apply extract_role function and overwrite Rota type column
    # full_appointments['Rota type'] = full_appointments['Rota type'].apply(extract_rota_type)
    ethnicity_dict = {
        "White": ["Other White", "British or Mixed British", "Irish"],
        "Black": ["African", "Other Black", "Caribbean"],
        "Mixed": [
            "Other Mixed",
            "White & Asian",
            "White & Black African",
            "White & Black Caribbean",
        ],
        "Asian": [
            "Other Asian",
            "Indian or British Indian",
            "Pakistani or British Pakistani",
            "Chinese",
            "Bangladeshi or British Bangladeshi",
        ],
        "Other": ["Other"],
    }

    for role, patterns in ethnicity_dict.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return role
    return "Unknown"
