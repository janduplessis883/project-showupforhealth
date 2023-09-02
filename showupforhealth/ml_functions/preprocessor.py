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
