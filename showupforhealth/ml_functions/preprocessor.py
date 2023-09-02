# Import Libriries 
import re
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder



# Jan pre-processor Functions
def format_datetime_columms(df):
    date_list = ['Appointment booked date','Appointment date','Registration date']
    for date in date_list:
        df[date] = pd.to_datetime(df[date])
    return df

def months_registered(df):
    # Calculate the difference between two dates
    df['delta'] = df['Appointment date'] - df['Registration date']
    
    # Convert Timedelta to months
    df['months_registered'] = df['delta'].dt.total_seconds() / (60*60*24*30.44)
    df['months_registered'] = np.ceil(df['months_registered'])
    
    # Drop the temporary 'delta' column
    df.drop(columns=['delta'], inplace=True)
    return df

def extract_rota_type(text):
    # HOW TO APPLY IT
    # Apply extract_role function and overwrite Rota type column
    # full_appointments['Rota type'] = full_appointments['Rota type'].apply(extract_rota_type)     
    role_map = {
    'GP': ['GP', 'Registrar', 'Urgent', 'Telephone', '111', 'FY2', 'F2', 'Extended Hours', 'GP Clinic', 'Session'],
    'Nurse': ['Nurse', 'Nurse Practitioner'], 
    'HCA': ['HCA','Health Care Assistant', 'Phlebotomy'],
    'ARRS': ['Pharmacist', 'Paramedic', 'Physiotherapist', 'Physicians Associate', 'ARRS', 'PCN'],
    }

    for role, patterns in role_map.items():
        for pattern in patterns:
            if re.search(pattern, text):
                return role
    return 'DROP'

def map_rota_type(df):
    df['Rota type'] = df['Rota type'].map(extract_rota_type)
    
    boolean_mask = (df['Rota type'] != 'DROP')
    # Applying the boolean filteraing
    df = df[boolean_mask].reset_index(drop=True)
    df.reset_index(inplace=True, drop=True)
    
    return df

# Label Encode Sex
def labelencode_sex(df):
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])
    
    return df