# Function for working on the appointment time to have 1 time only
def df_splitted():
    split_df = df['Appointment time'].str.split(' - ', expand=True)  # Split the timeframes
    df['Appointment time'] = split_df[0]  # Keep 'time1' and remove 'time2'
    df['app datetime'] = pd.to_datetime(df['Appointment date'] + ' ' + df['Appointment time'])
    return df


# Function merging the app datetime columns from two different csv files
def merged_weather():
    df_weather = pd.read_csv(WEATHER_DATA)
    df_weather['app datetime'] = pd.to_datetime(df_weather['app datetime'])
    merged_df = pd.merge(df, df_weather, on='app datetime')
    return merged_df

# testing

# Create a new column 'Booked_by_Gp' with 1 if booked by the same clinician, else 0
def booked_by():
    df['Booked_by_Gp'] = df.apply(lambda row: 1 if row['Clinician'] == row['Booked by'] else 0, axis=1)
    return df


# Function to hash patients IDs
import hashlib

def hash_patient_id(df, length=8):
    df['Patient ID'] = df['Patient ID'].apply(lambda x: hashlib.sha512(str(x).encode('utf-8')).hexdigest()[:length])
    return df

hash_patient_id(df)

from sklearn.preprocessing import OneHotEncoder

def one_hot_encode_columns(df, columns_to_encode=['Rota type', 'Ethnicity category', 'Language']):
    """
    Perform One-Hot Encoding on specified columns in a DataFrame.

    Parameters:
        df (pd.DataFrame): The input DataFrame.
        columns_to_encode (list): List of column names to OneHot Encode.

    Returns:
        pd.DataFrame: A new DataFrame with OneHot Encoded columns and original columns dropped.
    """
    # Create an instance of OneHotEncoder
    encoder = OneHotEncoder()

    # Initialize an empty DataFrame to store the encoded columns
    encoded_df = pd.DataFrame()

    # Loop over the columns to encode
    for column in columns_to_encode:
        # Fit the encoder to the current column and transform the data
        encoded_data = encoder.fit_transform(df[[column]])

        # Convert the result to a DataFrame and add it to the encoded DataFrame
        encoded_df = pd.concat([
            encoded_df,
            pd.DataFrame(encoded_data.toarray(), columns=encoder.get_feature_names_out([column]))
        ], axis=1)

    # Concatenate the encoded DataFrame with the original DataFrame, dropping the original columns
    df = pd.concat([df, encoded_df], axis=1)
    df.drop(columns=columns_to_encode, inplace=True)

    return df



