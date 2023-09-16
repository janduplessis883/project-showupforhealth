import streamlit as st
import joblib
import numpy as np
from sklearn.metrics import f1_score, custom_metric
from showupforhealth.params import *

from tensorflow import keras

@st.cache_data()
def predict_model():
    model = keras.models.load_model(filepath=MODEL_OUTPUT + '/model_weights_2023-09-12.h5', custom_objects={'f1_score': f1_score})
    return model
