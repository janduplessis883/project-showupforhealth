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