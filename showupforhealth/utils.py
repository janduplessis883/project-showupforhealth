from sklearn.model_selection import train_test_split

# X = df.drop('Appointment_status', axis=1)
# y = df['Appointment_status']

def perform_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test
