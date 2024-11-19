from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import pickle  # For loading the scaler
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import datetime  # For handling date input

app = Flask(__name__)

# Load the trained model
model = load_model(
    "saved_model/covid-India1_model.h5", custom_objects={"mse": MeanSquaredError()}
)

# Load the scaler used during training
scaler = pickle.load(open("saved_model/scaler.pkl", "rb"))


# Function to convert date to numerical format (days since a reference date)
def process_date(date_str):
    reference_date = datetime.date(2020, 1, 1)  # Reference start date
    input_date = datetime.date.fromisoformat(date_str)
    days_since_reference = (input_date - reference_date).days
    return days_since_reference


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

        # Process the date
        date = process_date(data["date"])

        # Collect features
        features = np.array(
            [
                date,
                data["daily_new_cases_avg"],
                data["active_cases"],
                data["cumulative_total_deaths"],
                data["daily_new_deaths_avg"],
                data["lagged_cumulative_cases_1"],
                data["lagged_cumulative_cases_2"],
                data["lagged_cumulative_cases_3"],
            ]
        ).reshape(1, -1)  # Reshape for scaler (1 row, 8 columns)

        # Normalize features
        normalized_features = scaler.transform(features)

        # Reshape for LSTM (1, sequence_length, num_features)
        lstm_input = normalized_features.reshape(
            1, normalized_features.shape[0], normalized_features.shape[1]
        )

        # Predict using the model
        prediction = model.predict(lstm_input).squeeze()

        # Inverse transform the prediction
        prediction = scaler.inverse_transform(
            [[0] * (features.shape[1] - 1) + [prediction]]
        )[0, -1]

        return jsonify({"predicted_cumulative_cases": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    app.run(debug=True)
