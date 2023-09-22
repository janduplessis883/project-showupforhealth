import pandas as pd 
import numpy as np
from colorama import Fore, Back, Style
from joblib import dump, load
from datetime import datetime

''' Scikit-Learn'''
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import BatchNormalization

''' Imbalanced Classes'''
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

''' Tensorflow Keras'''
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

from showupforhealth.utils import *
from showupforhealth.params import *
from showupforhealth.ml_functions.disease_register import *
from showupforhealth.ml_functions.data import *
from showupforhealth.ml_functions.encoders import *
from showupforhealth.ml_functions.preprocessor import *

from tensorflow.keras import backend as K

def f1_score(y_true, y_pred): # defining a custom F1 score metric
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

metrics = [
    keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
    f1_score,  # adding the custom F1 score metric
#     keras.metrics.TruePositives(name='tp'),
#     keras.metrics.FalsePositives(name='fp'),
#     keras.metrics.TrueNegatives(name='tn'),
#     keras.metrics.FalseNegatives(name='fn'), 
    keras.metrics.BinaryAccuracy(name='accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc'),
    keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]


def random_undersample(X_train, y_train, sampling_strategy=0.1):
    # Define undersampler
    rus = RandomUnderSampler(sampling_strategy=sampling_strategy)

    # Fit and transform the data
    X_trainu, y_trainu = rus.fit_resample(X_train, y_train)
    return X_trainu, y_trainu


def init_model():
    # Assuming X_train is globally accessible; otherwise, pass it as a parameter.
    # Only take the dimensions of a single sample, excluding the batch size.
    input_shape = X_train.shape[1:]

    model = models.Sequential()

    # Input layer specifying the shape
    model.add(layers.InputLayer(input_shape=input_shape))

    # First Dense layer of size 128
    model.add(layers.Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
        # First Dense layer of size 128
    model.add(layers.Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
        # First Dense layer of size 128
    model.add(layers.Dense(16, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    
        # First Dense layer of size 128
    model.add(layers.Dense(64, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))


    # Output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    optimizer = Adam(learning_rate=0.001)

    # Assuming 'metrics' is defined globally; otherwise, specify it directly or pass it as a parameter.
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=metrics)

    return model




if __name__ == '__main__':
    data = make_full_preprocess_data()
    df = feature_engineering(data)
    
    print(Fore.YELLOW + '\nDefine X & y')
    print(Style.RESET_ALL)
    X, y = define_X_y(df, 'Appointment_status')
    
    print(Fore.YELLOW + 'Drop Patient ID`')
    print(Style.RESET_ALL)
    X.drop(columns=['Patient ID'], inplace=True)
    
    print(Fore.YELLOW + 'Define X_train, X_val, X_test, y_train, y_val, y_test')
    print(Style.RESET_ALL)
    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y, val_size=0.2, test_size=0.1)
    
    print(Fore.YELLOW + 'Random Undersample with Imbalance')
    print(Style.RESET_ALL)
    X_train, y_train = random_undersample(X_train, y_train, sampling_strategy=0.1)

try:
    print(Fore.YELLOW + 'Scale and dump scaler to models folder Fit on X_train')
    print(Style.RESET_ALL)
    
    scaler = fit_scaler(X_train, scaler_type='standard')
    
    # Save scaler to Pickle for future use
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d %H-%M-%S")
    dump(scaler, f'{MODEL_OUTPUT}/scaler_{datetime_string}.pkl')
    
    print(Fore.YELLOW + '\nTransform X_train, X_val, and X_test')
    print(Style.RESET_ALL)
    
    X_train = transform_data(X_train, scaler)
    X_val = transform_data(X_val, scaler)
    X_test = transform_data(X_test, scaler)

except Exception as e:
    print(f"An error occurred: {str(e)}")

    
    print(Fore.YELLOW + 'Instantiate Model')
    print(Style.RESET_ALL)
    model = init_model()
    es = EarlyStopping(
    patience=30,
    monitor='val_recall', # We really want to detect fraudulent transactions!
    restore_best_weights=True)

    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=120,
                        batch_size=128, # Large enough to get a decent chance of containing fraudulent transactions 
                        callbacks=[es],
                        shuffle=True,
                        verbose=2
                    )
    
    model.predict(X_test)
    model.evaluate(X_test, y_test, verbose=0, return_dict=True)
    now = datetime.now()
    datetime_string = now.strftime("%Y-%m-%d %H-%M-%S")
    print(Fore.YELLOW + 'Save model to models folder')
    print(Style.RESET_ALL)
    model.save(f'{MODEL_OUTPUT}/model_weights_gentle_water{datetime_string}.h5') 