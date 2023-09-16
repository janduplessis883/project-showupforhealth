import streamlit as st
import joblib
import numpy as np

def predict_model(MODEL_OUTPUT, df):
  """Loads a model from a .h5 file and makes predictions on the given input data.

  Args:
    model_weight_file: The path to the .h5 file containing the model weights.
    input_data: A NumPy array containing the input data.

  Returns:
    A NumPy array containing the predicted output.
  """

  # Load the model.
  model = joblib.load(MODEL_OUTPUT)

  # Make predictions.
  predictions = model.predict(df)

  return predictions
