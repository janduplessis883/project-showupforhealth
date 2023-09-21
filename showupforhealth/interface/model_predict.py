import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from joblib import dump, load

import tensorflow as tf
from tensorflow.keras.models import load_model
from keras import backend as K
from keras.models import load_model
from colorama import Fore, Style

from showupforhealth.utils import *
from showupforhealth.params import *
from showupforhealth.ml_functions.predict import *

def f1_score(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val

def sort_df_columns(df):
    new_df = df[
        ['temp', 'precipitation', 'Age',
       'Sex', 'FRAILTY', 'DEPRESSION', 'OBESITY', 'IHD', 'DM', 'HPT', 'NDHG',
       'SMI', 'IMD2023', 'dist_to_station', 'distance_to_surg',
       'book_to_app_days', 'booked_by_clinician', 'registered_for_months',
       'sin_week', 'cos_week', 'sin_Appointment_time', 'cos_Appointment_time',
       'sin_month', 'cos_month', 'sin_day_of_week', 'cos_day_of_week',
       'No_shows', 'Rota_ARRS', 'Rota_GP', 'Rota_HCA', 'Rota_Nurse',
       'Ethnicity_Asian', 'Ethnicity_Black', 'Ethnicity_Mixed',
       'Ethnicity_Other', 'Ethnicity_White'
        ]
    ]
    return new_df

def scaler_model_predict(df):
    print(Fore.GREEN + '\n▶️ Select Scaler + Model:')
    print(Fore.GREEN + '1. jan_scaler_17sept23 + model16sept23_jan')
    print(Fore.GREEN + '2. Empty')
    print(Fore.GREEN + '3. Jan Model 17 Sept with Patient ID')
    scaler_no = input(Fore.RED + 'Enter Selection: ')
    print(Style.RESET_ALL)
    
    if scaler_no == '1':
        scaler = load(f'{MODEL_OUTPUT}/jan_scaler_17sept23.pkl')
        model = load_model(f'{MODEL_OUTPUT}/model16sept23_jan.h5', custom_objects={'f1_score': f1_score})
        
        df = sort_df_columns(df)
        df = transform_data(df, scaler)
        df = df.astype('float32')
        predictions = model.predict(df)
        return predictions
    
    elif scaler_no == '2':
        # Load scaler and model 2
        print('Note a valid saler + model selection.')
        pass
    
    elif scaler_no == '3':
        # Load scaler and model 3
        scaler = load(f'{MODEL_OUTPUT}/jan_scaler_17sept23withptid.pkl')
        model = load_model(f'{MODEL_OUTPUT}/model_weights_with_ptid2023-09-17 18-01-09.h5', custom_objects={'f1_score': f1_score})
        
        df = sort_df_columns(df)
        df = transform_data(df, scaler)
        df = df.astype('float32')
        predictions = model.predict(df)
        return predictions 
    
    else:
        print('Note a valid saler + model selection.')
        pass
        

def display_threshold_info(predictions, thresholds=[0.4, 0.5, 0.6, 0.7, 0.8]):
    for t in thresholds:
        class_labels = (predictions > t).astype(int)
        print(f'No shows Predicted at {t} threshold: {class_labels.flatten().tolist().count(0)}')

    select_threshold = input(Fore.RED + 'Select Threshold to continue: ')
    print(Style.RESET_ALL)
    class_labels = (predictions > float(select_threshold)).astype(int)
    
    return class_labels

def display_outcome_df(class_labels, pt_id_df):
    class_labels_list = class_labels.flatten().tolist()
    new_col = pd.Series(class_labels_list, name='Model_Prediction')

    df = pd.concat([pt_id_df, new_col], axis=1)

    surgery = pd.read_csv(f'{PREDICT_DATA}/original/{surgery_prefix}_Predict.csv')
    merged = surgery.merge(df, how='left', on="Patient ID")
    merged_short = merged[['Appointment status','Model_Prediction', 'Patient ID']]
    filtered_df = merged_short[merged_short['Model_Prediction'] == 0]
    print(f'Surgery: {surgery_prefix}')
    return filtered_df




if __name__ == "__main__":
    print(Fore.BLUE + '\nECS - Earls Court Surgery')
    print(Fore.BLUE + 'TGP - The Good Practice')
    print(Fore.BLUE + 'TCP - The Chelsea Practice')
    print(Fore.BLUE + 'SMW - Stanhope Mews West')
    print(Fore.BLUE + 'KMC - Knightsbridge Medical Centre')
    print(Fore.BLUE + 'HPVN - Health Partners at Violet Melchett')
    surgery_prefix = input(Fore.RED + 'Enter Surgery Prefix to continue: ')   
    print(Style.RESET_ALL)
    
    X_temp = streamlit_predict(surgery_prefix)
    X_new = X_temp.drop(columns='Patient ID')
    pt_id = X_temp[["Patient ID"]]

    X_new.shape
    
    # Load Scaler
    predictions = scaler_model_predict(X_new)
    class_labels = display_threshold_info(predictions)
    df = display_outcome_df(class_labels, pt_id)
    display(df)
    display(df['Appointment status'].value_counts())