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

def split_appointment_date(df):
    # Convert the "Appointment date" column to datetime if it's not already
    df['Appointment date'] = pd.to_datetime(df['Appointment date'], format='%d-%b-%y')

    # Extract day, week, and month
    df['Day'] = df['Appointment date'].dt.dayofweek  # 0-6 (Monday to Sunday)
    df['Week'] = df['Appointment date'].dt.isocalendar().week  # 1-52
    df['Month'] = df['Appointment date'].dt.month  # 1-12 (January to December)

    return df 

 def filter_current_registration(df):
    # Filter rows where 'Registration status' is 'Current'
    data = df[df['Registration status'] == 'Current']
    
    return df

#calculate BMI and encode depression
def process_dataframe(df):
    # Check if HEIGHT and WEIGHT columns have been cleaned
    if not df['WEIGHT'].dtype == float:
        # Clean the WEIGHT column
        def clean_weight(weight_str):
            if isinstance(weight_str, str):
                match = re.search(r'\d+', weight_str)
                if match:
                    return float(match.group())
            return None

        df['WEIGHT'] = df['WEIGHT'].apply(clean_weight)

    if not df['HEIGHT'].dtype == float:
        # Clean the HEIGHT column
        def clean_height(height_str):
            if isinstance(height_str, str):
                match = re.search(r'\d+\.\d+', height_str)
                if match:
                    return float(match.group())
            return None

        df['HEIGHT'] = df['HEIGHT'].apply(clean_height)

    # Calculate BMI
    df['BMI'] = (df['WEIGHT'] / (df['HEIGHT'] ** 2))

    # Create the 'depr' column
    df['depr'] = df['Depression'].apply(lambda x: 1 if isinstance(x, str) and re.search(r'\S', x) else 0)

    return df

# days until appointment function
def calculate_days_difference(df):
    # Convert the date columns to datetime objects
    df['Appointment booked date'] = pd.to_datetime(df['Appointment booked date'], format='%d-%b-%y')
    df['Appointment date'] = pd.to_datetime(df['Appointment date'], format='%d-%b-%y')

    # Calculate the difference in days and create a new column
    df['Days Difference'] = (df['Appointment date'] - df['Appointment booked date']).dt.days
    
    # Drop rows where 'Days Difference' is negative
    df = df[df['Days Difference'] >= 0]

    return df