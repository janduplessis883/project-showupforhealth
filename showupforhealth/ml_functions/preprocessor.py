from showupforhealth import params
import pandas as pd


# Function for working on the appointment time to have 1 time only
def df_splitted(df):
    split_df = df['Appointment time'].str.split(' - ', expand=True)  # Split the timeframes
    df['Appointment time'] = split_df[0]  # Keep 'time1' and remove 'time2'
    df['app datetime'] = pd.to_datetime(df['Appointment date'] + ' ' + df['Appointment time'])
    return df


# Function merging the app datetime columns from two different csv files
def merged_weather(df):
    df_weather = pd.read_csv(WEATHER_DATA)
    df_weather['app datetime'] = pd.to_datetime(df_weather['app datetime'])
    merged_df = pd.merge(df, df_weather, on='app datetime')
    return merged_df

# testing

# Create a new column 'Booked_by_Gp' with 1 if booked by the same clinician, else 0
def booked_by(df):
    df['Booked_by_Gp'] = df.apply(lambda row: 1 if row['Clinician'] == row['Booked by'] else 0, axis=1)
    return df

import hashlib

# Function to hash values using SHA-256 and truncate the result
def hash_and_truncate(value, length=8):
    # Convert the value to a string
    value_str = str(value)

    # Create a hash object using SHA-256
    sha256 = hashlib.sha256()

    # Update the hash object with the value
    sha256.update(value_str.encode('utf-8'))

    # Get the hexadecimal representation of the hash and truncate it
    hashed_value = sha256.hexdigest()[:length]

    return hashed_value

# Apply the hash function to the 'Patient_ID' column and create a new 'Hashed_Patient_ID' column
df['Hashed Patient ID'] = df['Patient ID'].apply(hash_and_truncate)

# Drop the original 'Patient_ID' column if you no longer need it
df.drop(columns=['Patient ID'], inplace=True)


from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_columns(df, columns_to_encode):
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

# Columns to One-Hot Encode
columns_to_encode = ['Rota type', 'Ethnicity category', 'Language']

# Call the function to perform One-Hot Encoding
df_encoded = one_hot_encode_columns(df, columns_to_encode)

# Import Libriries 
import re
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def encode_appointment_status(df):
    df['App_status_encoded'] = [0 if app_status == 'Did Not Attend' else 1 for app_status in df['Appointment status']]
    df = df.drop(columns='Appointment status')
    return df

def encode_hour_appointment(df):
    df['Hour_of_appointment'] = df['Appointment time'].str[:2].astype(int)
    df.drop(columns=['Appointment time'], inplace=True)
    return df

def group_ethnicity_categories(df):
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
    df = df.rename(columns={'Ethnicity category': 'Ethnicity'})
    ohe = OneHotEncoder(sparse_output=False)
    ohe.fit(df[['Ethnicity']])
    df[ohe.get_feature_names_out()] = ohe.transform(df[['Ethnicity']])
    df.drop(columns=['Ethnicity'], inplace=True)
    return df

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





    
    