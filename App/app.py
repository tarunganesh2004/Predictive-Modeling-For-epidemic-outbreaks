from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle  # For loading the scaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError

app = Flask(__name__)

# Load the trained model
model = load_model(
    "saved_model/covid-India1_model.h5", custom_objects={"mse": MeanSquaredError()}
)

# Load the scaler used during training
scaler = pickle.load(open("saved_model/scaler.pkl", "rb"))


# Route for the main page
@app.route("/")
def index():
    return render_template("index.html")


# API endpoint for predictions
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get the input JSON data
        data = request.get_json()

        # Ensure features are in the correct shape
        features = np.array(
            [
                data["date"],
                data["daily_new_cases_avg"],
                data["active_cases"],
                data["cumulative_total_deaths"],
                data["daily_new_deaths_avg"],
                data["lagged_cumulative_cases_1"],
                data["lagged_cumulative_cases_2"],
                data["lagged_cumulative_cases_3"],
            ]
        ).reshape(1, -1)  # Shape (1, 8)

        # Scale the features using the scaler
        scaled_features = scaler.transform(features)

        # Reshape to (1, seq_length, num_features) for the LSTM
        scaled_features = scaled_features.reshape(1, 1, -1)

        # Predict using the model
        prediction = model.predict(scaled_features).squeeze()

        # Inverse transform the prediction (only the target variable, cumulative cases)
        prediction = scaler.inverse_transform(
            [[0] * (features.shape[1] - 1) + [prediction]]
        )[0, -1]

        return jsonify({"predicted_cumulative_cases": float(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
