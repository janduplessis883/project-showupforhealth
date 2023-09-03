import pandas as pd

from showupforhealth.params import *
from showupforhealth.ml_functions.disease_register import make_global_disease_register


def create_global_appointments_list(surgery_list = ['ECS', 'TCP', 'TGP', 'SMW', 'KMC', 'HPVM']):
    print("\n‼️ Processing Appointments data ======================================================")
    full_app_list = []
    for surgery_prefix in surgery_list:
        print(f'⏺️: {surgery_prefix} -', end=' ')
        df_list = []
        for i in range(1,10,1):
            app = pd.read_csv(f'{RAW_DATA}{surgery_prefix}/{surgery_prefix}_APP{i}.csv')
            print(f"df {i} ", end=' ')
            df_list.append(app)

        appointments = pd.concat(df_list, axis=0, ignore_index=True)
        print(f'duplicates: {appointments.duplicated().sum()}')
        full_app_list.append(appointments)
        
        global_appointments = pd.concat(full_app_list, axis=0, ignore_index=True)
        
    print(f'✅ Appointment List - {global_appointments.shape}')
    # Filter and drop rows with 'DROP' value
    return global_appointments



def add_weather(global_apps):     
    weather = pd.read_csv(WEATHER_DATA)
    
    weather['app datetime'] = pd.to_datetime(weather['app datetime'])


    global_apps['app datetime'] = pd.to_datetime(global_apps['Appointment date'] + ' ' +
                                        global_apps['Appointment time'].str.split(expand=True)[0])


    global_apps['app datetime'] = pd.to_datetime(global_apps['app datetime'])

    global_apps_weather = global_apps.merge(weather, how='left', on='app datetime')
    print(f'🌤️ Weather Added to Apps {global_apps_weather.shape}')
    return global_apps_weather

def add_disease_register():
    disease_path = f'{OUTPUT_DATA}global_disease_register.csv'
    disease = pd.read_csv(disease_path)
    
    apps_path = f'{OUTPUT_DATA}global_apps_list.csv'
    global_apps = pd.read_csv(apps_path)

    raw_train_data = global_apps.merge(disease, how='left', on='Patient ID')

    print(f'😷 Disease Register Added to Apps {raw_train_data.shape} - saved as full_raw_train_data.csv')
    return raw_train_data


def make_full_preprocess_data():
    register = make_global_disease_register()
    register['Patient ID'] = register['Patient ID'].astype('int64')
    apps = create_global_appointments_list()
    
    apps_weather = add_weather(apps)
    apps_weather['Patient ID'] = apps_weather['Patient ID'].astype('int64')
    full_df = apps_weather.merge(register, how='left', on='Patient ID')
    print(f'↔️ Merged Appointments and Global Register - Pre-process df {full_df.shape}')
    full_df.dropna(inplace=True)
    print(f'❌ dropna {full_df.shape}')
    full_path = f'{OUTPUT_DATA}full_preprocess_data.csv'
    
    print(f'💾 Saving to output_data/full_preprocess_data.csv...')
    full_df.to_csv(full_path, index=False)
    print('✅ Done')
    return full_df

 
if __name__ == '__main__':
    make_full_preprocess_data()