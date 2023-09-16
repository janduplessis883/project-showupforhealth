import streamlit as st
import joblib
import numpy as np
from showupforhealth.params import *


import keras

@st.cache(allow_output_mutation=True)
def predict_model():
    model = keras.models.load_model(MODEL_OUTPUT)
    return model
