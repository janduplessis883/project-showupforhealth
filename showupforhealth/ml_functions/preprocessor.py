# Import Libriries
import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from showupforhealth import params
import pandas as pd
import hashlib
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

from showupforhealth.params import *


# Create a new column 'Booked_by_Gp' with 1 if booked by the same clinician, else 0
def booked_by_clinicain(df):
    print('➡️ Booked by Clinician')
    df['booked_by_clinician'] = df.apply(lambda row: 1 if row['Clinician'] == row['Booked by'] else 0, axis=1)
    df.drop(columns=['Booked by','Clinician', 'NHS number', 'app datetime'], inplace=True)
    return df


def hash_patient_id(df, length=8):
    print('➡️ Hash Patient ID')
    df['Patient ID'] = df['Patient ID'].apply(lambda x: hashlib.sha512(str(x).encode('utf-8')).hexdigest()[:length])
    return df


# # Function to hash values using SHA-256 and truncate the result
# def hash_and_truncate(value, length=8):
#     # Convert the value to a string
#     value_str = str(value)

#     # Create a hash object using SHA-256
#     sha256 = hashlib.sha256()

#     # Update the hash object with the value
#     sha256.update(value_str.encode('utf-8'))

#     # Get the hexadecimal representation of the hash and truncate it
#     hashed_value = sha256.hexdigest()[:length]

#     return hashed_value

# # Apply the hash function to the 'Patient_ID' column and create a new 'Hashed_Patient_ID' column
# df['Hashed Patient ID'] = df['Patient ID'].apply(hash_and_truncate)

# # Drop the original 'Patient_ID' column if you no longer need it
# df.drop(columns=['Patient ID'], inplace=True)




def one_hot_encode_columns(df, columns_to_encode=['Rota']):
    print('➡️ OHE Columns')
    """
    Perform One-Hot Encoding on specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (list): List of column names to One-Hot Encode.

    Returns:
        pd.DataFrame: A new DataFrame with One-Hot Encoded columns and original columns dropped.
    """
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # Fit the encoder to the specified categorical columns and transform the data
    encoded_data = encoder.fit_transform(df[columns_to_encode])

    # Convert the result to a DataFrame
    encoded_df = pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out(columns_to_encode))

    # Concatenate the encoded DataFrame with the original DataFrame, dropping the original columns
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=columns_to_encode, inplace=True)

    return df

# mtesting 

def encode_appointment_status(df):
    print('➡️ Encode Appointment status')
    df['Appointment_status'] = [0 if app_status == 'Did Not Attend' else 1 for app_status in df['Appointment status']]
    df = df.drop(columns='Appointment status')
    return df

def encode_hour_appointment(df):
    print('➡️ Extract Time of Appointment')
    df['hour_of_appointment'] = df['Appointment time'].str[:2].astype(int)
    df.drop(columns=['Appointment time'], inplace=True)
    return df

def group_ethnicity_categories(df):
    print('➡️ Mapping Ethnicity category')
    ethnicity_dict = {'African': 'Black',
                      'Other Black': 'Black',
                      'Caribbean': 'Black',
                      'British or Mixed British': 'White',
                      'Other White': 'White',
                      'Irish': 'White',
                      'White & Black African': 'Mixed',
                      'White & Black Caribbean': 'Mixed',
                      'White & Asian': 'Mixed',
                      'Other Mixed': 'Mixed',
                      'Other Asian': 'Asian',
                      'Indian or British Indian': 'Asian',
                      'Pakistani or British Pakistani': 'Asian',
                      'Chinese': 'Asian',
                      'Bangladeshi or British Bangladeshi': 'Asian',
                      'Other': 'Unknown'}
    df['Ethnicity category'] = df['Ethnicity category'].map(ethnicity_dict)
    return df

def ohe_ethnicity(df):
    print('➡️ OHE Ethnicity')
    df = df.rename(columns={'Ethnicity category': 'Ethnicity'})
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(df[['Ethnicity']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['Ethnicity']])
    df.drop(columns=['Ethnicity'], inplace=True)
    return df

# Jan pre-processor Functions
def format_datetime_columms(df):
    print('➡️ Convert Datetime Columns')
    date_list = ['Appointment booked date','Appointment date','Registration date']
    for date in date_list:
        df[date] = pd.to_datetime(df[date])
    return df

def months_registered(df):
    print('➡️ Months Registered with practice')
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
    print('➡️ Map Rota types - renamed Rota')
    df['Rota type'] = df['Rota type'].map(extract_rota_type)
    
    df.drop(df[df['Rota type'] == 'DROP'].index, inplace=True)

    # Rename column
    df.rename(columns={'Rota type':'Rota'}, inplace=True)

    return df

# Label Encode Sex
def labelencode_sex(df):
    print('➡️ Labelencoding Sex')
    le = LabelEncoder()
    df['Sex'] = le.fit_transform(df['Sex'])

    return df

def split_appointment_date(df):
    print('➡️ Split appointment date')
    # Convert the "Appointment date" column to datetime if it's not already
    df['Appointment date'] = pd.to_datetime(df['Appointment date'], format='%d-%b-%y')

    # Extract day, week, and month
    df['Day'] = df['Appointment date'].dt.dayofweek  # 0-6 (Monday to Sunday)
    df['Week'] = df['Appointment date'].dt.isocalendar().week  # 1-52
    df['Month'] = df['Appointment date'].dt.month  # 1-12 (January to December)

    return df

def filter_current_registration(df):
    print('➡️ Deseased and deducted')
    # Filter rows where 'Registration status' is 'Current'
    df = df[df['Registration status'] == 'Current']

    return df


# days until appointment function
def calculate_days_difference(df):
    print('➡️ Months Registered with practice')
    # Convert the date columns to datetime objects
    df['Appointment booked date'] = pd.to_datetime(df['Appointment booked date'], format='%d-%b-%y')
    df['Appointment date'] = pd.to_datetime(df['Appointment date'], format='%d-%b-%y')

    # Calculate the difference in days and create a new column
    df['days_booked_to_app'] = (df['Appointment date'] - df['Appointment booked date']).dt.days

    # Drop rows where 'Days Difference' is negative
    df = df[df['days_booked_to_app'] >= 0]

    return df

def drop_rename_columns(df):
    print('➡️ Drop and rename columns')
    df.drop(columns=['Postcode','Latitude','Longitude', 'Appointment booked date', 'Appointment status', 'Registration date'], inplace=True)
    df.rename(columns={'Age in years': 'Age'}, inplace=True)
    return df

def feature_engeneering(df):
    print('‼️ Feature Engineering =================================================================')
    format_datetime_columms(df)
    booked_by_clinicain(df)
    hash_patient_id(df)
    encode_appointment_status(df)
    group_ethnicity_categories(df)
    months_registered(df)
    map_rota_type(df)
    labelencode_sex(df)
    encode_hour_appointment(df)
    split_appointment_date(df)
    filter_current_registration(df)
    calculate_days_difference(df)
    one_hot_encode_columns(df)
    drop_rename_columns(df)
    return df





    
    
