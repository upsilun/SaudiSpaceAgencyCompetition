import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import OneClassSVM
import joblib

def generate_synthetic_data(num_samples=1000):
    """
    Generates a DataFrame with synthetic satellite data for training purposes.
    This function creates data that mimics real-world sensor readings and outcomes.
    """
    # Seed for reproducibility
    np.random.seed(42)

    # --- Generate data for each feature based on plausible ranges ---
    data = {
        # Power System Data
        'battery_voltage': np.random.uniform(12.0, 14.0, num_samples),
        'battery_temperature': np.random.uniform(20.0, 40.0, num_samples),
        'solar_panel_current': np.random.uniform(1.0, 5.0, num_samples),
        'cycle_count': np.arange(num_samples),

        # Propulsion System Data
        'thruster_capacitor_temp': np.random.uniform(50.0, 70.0, num_samples),
        'thruster_pulse_count': np.arange(num_samples) * 10,
        'servo_current': np.random.uniform(0.1, 0.5, num_samples),
        'servo_position_error': np.random.uniform(-0.1, 0.1, num_samples),

        # Guidance & Control System Data
        'gyro_drift_rate': np.random.uniform(0.001, 0.01, num_samples),
        'processor_temperature': np.random.uniform(60.0, 85.0, num_samples),
        'processor_load': np.random.uniform(0.2, 0.8, num_samples),

        # Environmental Data
        'module_temperature': np.random.uniform(15.0, 35.0, num_samples),
        'radiation_dose': np.random.uniform(0.0, 0.1, num_samples) * np.arange(num_samples)
    }
    df = pd.DataFrame(data)

    # --- Create Target Variables based on the generated features ---

    # 1. Remaining Useful Life (RUL)
    # RUL decreases as cycle count and temperature increase.
    df['rul'] = 365 - (df['cycle_count'] * 0.1) - (df['battery_temperature'] - 20) * 0.5 - (df['processor_temperature'] - 60) * 0.3
    df['rul'] = df['rul'].clip(lower=0) # Ensure RUL doesn't go below zero

    # 2. System Health Status
    # Status is determined by voltage, temperature, and errors.
    conditions = [
        (df['battery_voltage'] < 12.2) | (df['processor_temperature'] > 80),
        (df['battery_voltage'] < 12.8) | (df['processor_temperature'] > 75),
        (df['battery_voltage'] >= 12.8)
    ]
    choices = ['Critical', 'Warning', 'Nominal']
    df['status'] = np.select(conditions, choices, default='Nominal')

    # 3. Anomaly Detection
    # Introduce some anomalies for the model to learn.
    df.loc[df.sample(frac=0.05).index, 'battery_voltage'] *= 0.8 # Sudden voltage drop
    df.loc[df.sample(frac=0.05).index, 'servo_position_error'] *= 5 # Large position error
    # Label for anomaly detection (1 for normal, -1 for anomaly) - used by OneClassSVM
    df['anomaly'] = 1
    df.loc[df['battery_voltage'] < 11.5, 'anomaly'] = -1
    df.loc[df['servo_position_error'] > 0.3, 'anomaly'] = -1


    return df

def train_models(df):
    """
    Trains the AI models on the provided DataFrame and saves them to disk.
    """
    # --- Feature and Target Selection ---
    # Define which columns are input features and which are the targets to predict.
    features = [
        'battery_voltage', 'battery_temperature', 'solar_panel_current', 'cycle_count',
        'thruster_capacitor_temp', 'thruster_pulse_count', 'servo_current',
        'servo_position_error', 'gyro_drift_rate', 'processor_temperature',
        'processor_load', 'module_temperature', 'radiation_dose'
    ]
    X = df[features]
    y_rul = df['rul']
    y_status = df['status']
    X_anomaly = df[df['anomaly'] == 1][features] # Train anomaly detector only on normal data

    # --- Data Preprocessing ---
    # Split data for RUL and Status models
    X_train, X_test, y_rul_train, y_rul_test, y_status_train, y_status_test = train_test_split(
        X, y_rul, y_status, test_size=0.2, random_state=42)

    # Scale the features to have zero mean and unit variance. This is important for many models.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    # Note: We use the same scaler to transform the anomaly data.
    X_anomaly_scaled = scaler.transform(X_anomaly)

    # --- 1. Train RUL Prediction Model (Linear Regression) ---
    print("Training Remaining Useful Life (RUL) model...")
    rul_model = LinearRegression()
    rul_model.fit(X_train_scaled, y_rul_train)
    print("RUL model training complete.")

    # --- 2. Train System Health Status Model (Random Forest) ---
    print("Training System Health Status model...")
    status_model = RandomForestClassifier(n_estimators=100, random_state=42)
    status_model.fit(X_train_scaled, y_status_train)
    print("Status model training complete.")

    # --- 3. Train Anomaly Detection Model (One-Class SVM) ---
    print("Training Anomaly Detection model...")
    anomaly_model = OneClassSVM(nu=0.05, kernel='rbf', gamma='auto') # nu is the expected proportion of anomalies
    anomaly_model.fit(X_anomaly_scaled)
    # We rename the output for clarity in the API
    anomaly_model.classes_ = np.array(['Anomaly Detected', 'Nominal'])
    print("Anomaly model training complete.")


    # --- Save the Models and the Scaler ---
    # The models and scaler are saved to files for later use by the API.
    print("Saving models to disk...")
    joblib.dump(rul_model, 'rul_model.pkl')
    joblib.dump(status_model, 'status_model.pkl')
    joblib.dump(anomaly_model, 'anomaly_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    print("Models and scaler saved successfully.")

# --- Main script execution ---
if __name__ == '__main__':
    print("Generating synthetic data...")
    # Create the dataset
    synthetic_df = generate_synthetic_data(2000)
    print("Data generation complete.")
    # Train the models on the dataset
    train_models(synthetic_df)
    print("\nTraining process finished. You can now run 'satellite_api.py'.")
