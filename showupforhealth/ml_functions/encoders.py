import math
import pandas as pd

def haversine_distance(surgery_prefix, lat2, lon2):
    R = 6371.0  # Radius of the Earth in kilometers

    if surgery_prefix == 'ECS':
        lat1, lon1 = 51.488721, -0.191873
    elif surgery_prefix == 'SMW':
        lat1, lon1 = 51.494474, -0.181931
    elif surgery_prefix == 'TCP':
        lat1, lon1 = 51.48459, -0.171887
    elif surgery_prefix == 'HPVM':
        lat1, lon1 = 51.48459, -0.171887
    elif surgery_prefix == 'KMC':
        lat1, lon1 = 51.49807, -0.159918
    elif surgery_prefix == 'TGP':
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
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance  # in kilometers