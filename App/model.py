import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

# Load the model with mse as the loss function
model = load_model("covid-India1_model.h5", custom_objects={"mse": MeanSquaredError()})


def load_model():
    return model
#     # Load the saved model
#     return tf.keras.models.load_model("covid-India1_model.h5")


def predict(data, model):
    # Process the input data and make prediction
    data = np.array(data).reshape(1, -1)  # Adjust based on input format
    prediction = model.predict(data)
    return prediction.tolist()
