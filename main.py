import flask
from flask import request, jsonify
import joblib

# Initialize the Flask application
app = flask.Flask(__name__)

# --- Load the Trained Models and Scaler ---
# It's expected that these files are in the same directory as the script.
# These files are created by the 'train_ai.py' script.
try:
    # Load the model that predicts Remaining Useful Life (RUL)
    rul_model = joblib.load('rul_model.pkl')
    # Load the model that classifies the system's health status
    status_model = joblib.load('status_model.pkl')
    # Load the model that detects anomalies
    anomaly_model = joblib.load('anomaly_model.pkl')
    # Load the scaler object which is used to normalize input data
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    # If any model file is not found, print an error and exit.
    # This prevents the API from running without the necessary components.
    print("Error: Model files not found. Please run 'train_ai.py' first.")
    exit()


# --- Define the Prediction API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    """
    This function handles the prediction requests.
    It receives data, preprocesses it, and uses the loaded models to make predictions.
    """
    # Get the JSON data from the request
    data = request.get_json(force=True)

    # --- Prepare the input data for the models ---
    # The order of features must be the same as used during training.
    features = [
        data['battery_voltage'],
        data['battery_temperature'],
        data['solar_panel_current'],
        data['cycle_count'],
        data['thruster_capacitor_temp'],
        data['thruster_pulse_count'],
        data['servo_current'],
        data['servo_position_error'],
        data['gyro_drift_rate'],
        data['processor_temperature'],
        data['processor_load'],
        data['module_temperature'],
        data['radiation_dose']
    ]

    # Scale the features using the loaded scaler
    scaled_features = scaler.transform([features])

    # --- Make Predictions Using the Loaded Models ---
    rul_prediction = rul_model.predict(scaled_features)[0]
    status_prediction = status_model.predict(scaled_features)[0]
    # OneClassSVM returns 1 for normal, -1 for anomaly. These can be numpy.int64.
    anomaly_raw_prediction = anomaly_model.predict(scaled_features)[0]

    # Map the integer output from the anomaly model to a human-readable string
    anomaly_prediction = 'Nominal' if anomaly_raw_prediction == 1 else 'Anomaly Detected'


    # --- Determine the Optimal Action ---
    # This is a simplified logic for recommending an action.
    # In a real-world scenario, this would be a more complex decision-making model.
    if status_prediction == 'Critical' or rul_prediction < 30: # Assuming RUL is in days
        action = "Initiate De-orbit Burn"
    elif status_prediction == 'Warning' or anomaly_prediction == 'Anomaly Detected':
        action = "Enter Safe Mode and Alert Ground Control"
    else:
        action = "Nominal Operations"

    # --- Format and Return the Response ---
    # Explicitly cast all values to standard Python types to ensure they are JSON serializable
    response = {
        'remaining_useful_life_days': float(round(rul_prediction, 2)),
        'system_health_status': str(status_prediction),
        'anomaly_detection_flag': str(anomaly_prediction),
        'optimal_action_recommendation': str(action)
    }

    return jsonify(response)


# --- Main entry point to run the Flask app ---
if __name__ == '__main__':
    # Runs the app on the local development server.
    # The host '0.0.0.0' makes the server accessible from any IP address.
    app.run(port=5000, debug=True, host='0.0.0.0')

