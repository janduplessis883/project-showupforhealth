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
