from sklearn.model_selection import train_test_split

# X = df.drop('Appointment_status', axis=1)
# y = df['Appointment_status']

def perform_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test

from imblearn.over_sampling import SMOTE
import numpy as np

def oversample_with_smote(X_train, y_train, sampling_strategy='auto', k_neighbors=5, random_state=42):
    smote = SMOTE(sampling_strategy=sampling_strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_train_oversampled, y_train_oversampled = smote.fit_resample(X_train, y_train)
    print('Shape of X_train:', np.shape(X_train_oversampled))
    print('Shape of y_train:', np.shape(y_train_oversampled))
    return X_train_oversampled, y_train_oversampled


