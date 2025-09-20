# **Satellite Predictive Maintenance AI**

## **1\. Overview**

This project implements a simple predictive maintenance AI for a satellite, exposed via a Flask API. The AI analyzes simulated telemetry data from various satellite subsystems to predict its Remaining Useful Life (RUL), assess its overall health, detect anomalies, and recommend optimal actions (such as entering safe mode or initiating a de-orbit burn).  
The primary goal is to provide an intelligent, autonomous decision-making layer that can help ensure mission success and responsible satellite operation.

## **2\. Features**

The AI provides four key outputs based on the input telemetry data:

* **Remaining Useful Life (RUL) Prediction:** Estimates the operational lifespan of key components (e.g., "Battery RUL: 90 days").  
* **System Health Status Classification:** Provides a simple status: Nominal, Warning, or Critical.  
* **Anomaly Detection Flag:** Identifies unexpected behavior that deviates from normal patterns.  
* **Optimal Action Recommendation:** Suggests a high-level command like "Initiate De-orbit Burn" or "Enter Safe Mode" to maximize mission success probability.

## **3\. Technology Stack**

* **Backend:** Python  
* **API Framework:** Flask  
* **Machine Learning:** Scikit-learn  
* **Data Handling:** Pandas, NumPy

## **4\. Setup and Usage**

Follow these steps to get the project running on your local machine.

### **Step 1: Prerequisites**

* Make sure you have **Python 3.7** or newer installed.

### **Step 2: Install Dependencies**

Open your terminal or command prompt and run the following command to install the required Python libraries:  
pip install pandas scikit-learn flask

### **Step 3: Run the API Server**

Once the models are trained and saved, start the Flask API server:  
python main.py

You should see an output indicating that the server is running, typically on http://127.0.0.1:5000.

### **Step 4: Test the API**

You can now send POST requests to the API. Use a tool like **Postman** or cURL. The test cases below provide detailed examples.

## **5\. Test Cases and Scenarios**

Here are several scenarios with corresponding JSON payloads to test the AI's response under different conditions.

### **Scenario 1: Optimal Health (Beginning of Life)**

Description: The satellite has just been deployed. All systems are new and telemetry readings are perfect.  
Expected AI Response: Nominal status, high RUL, and "Nominal Operations" recommendation.  
```
{  
    "battery\_voltage": 14.0,  
    "battery\_temperature": 25.0,  
    "solar\_panel\_current": 5.0,  
    "cycle\_count": 10,  
    "thruster\_capacitor\_temp": 55.0,  
    "thruster\_pulse\_count": 100,  
    "servo\_current": 0.18,  
    "servo\_position\_error": 0.001,  
    "gyro\_drift\_rate": 0.001,  
    "processor\_temperature": 60.0,  
    "processor\_load": 0.3,  
    "module\_temperature": 22.0,  
    "radiation\_dose": 5.0  
}
```
### **Scenario 2: Normal Wear and Tear (Mid-Life)**

Description: The satellite has been in orbit for some time. Minor degradation is visible in the battery and cycle counts are higher, but all systems are still well within safe operating limits.  
Expected AI Response: Nominal status, slightly reduced RUL, "Nominal Operations" recommendation.  
```
{  
    "battery\_voltage": 13.6,  
    "battery\_temperature": 29.0,  
    "solar\_panel\_current": 4.3,  
    "cycle\_count": 600,  
    "thruster\_capacitor\_temp": 64.0,  
    "thruster\_pulse\_count": 6000,  
    "servo\_current": 0.22,  
    "servo\_position\_error": 0.03,  
    "gyro\_drift\_rate": 0.004,  
    "processor\_temperature": 70.0,  
    "processor\_load": 0.55,  
    "module\_temperature": 26.0,  
    "radiation\_dose": 45.0  
}
```
### **Scenario 3: Developing Issue (Warning)**

Description: The battery voltage is consistently low and its temperature is high, indicating accelerated degradation. The RUL is dropping faster than expected.  
Expected AI Response: Warning status, lower RUL, and a recommendation to "Enter Safe Mode".  
```
{  
    "battery\_voltage": 12.8,  
    "battery\_temperature": 36.0,  
    "solar\_panel\_current": 3.0,  
    "cycle\_count": 1200,  
    "thruster\_capacitor\_temp": 69.0,  
    "thruster\_pulse\_count": 11500,  
    "servo\_current": 0.28,  
    "servo\_position\_error": 0.09,  
    "gyro\_drift\_rate": 0.008,  
    "processor\_temperature": 78.0,  
    "processor\_load": 0.7,  
    "module\_temperature": 31.0,  
    "radiation\_dose": 80.0  
}
```
### **Scenario 4: Anomaly Event (Warning)**

Description: A sudden, unexpected spike in the gyroscope drift rate has occurred. While not immediately critical, it deviates from the norm and could indicate a control system fault.  
Expected AI Response: Anomaly Detected flag, Warning status, and recommendation to "Enter Safe Mode and Alert Ground Control".  
```
{  
    "battery\_voltage": 13.5,  
    "battery\_temperature": 30.0,  
    "solar\_panel\_current": 4.2,  
    "cycle\_count": 900,  
    "thruster\_capacitor\_temp": 65.0,  
    "thruster\_pulse\_count": 9000,  
    "servo\_current": 0.23,  
    "servo\_position\_error": 0.04,  
    "gyro\_drift\_rate": 0.15,  
    "processor\_temperature": 71.0,  
    "processor\_load": 0.6,  
    "module\_temperature": 27.0,  
    "radiation\_dose": 65.0  
}
```
### **Scenario 5: Imminent Failure (Critical)**

Description: Multiple indicators are critical. The battery is failing, the processor is overheating, and radiation dose is high. The predicted RUL is very low.  
Expected AI Response: Critical status, very low RUL, and an urgent recommendation to "Initiate De-orbit Burn".  
```
{  
    "battery\_voltage": 12.1,  
    "battery\_temperature": 40.0,  
    "solar\_panel\_current": 2.1,  
    "cycle\_count": 2000,  
    "thruster\_capacitor\_temp": 75.0,  
    "thruster\_pulse\_count": 22000,  
    "servo\_current": 0.5,  
    "servo\_position\_error": 0.2,  
    "gyro\_drift\_rate": 0.02,  
    "processor\_temperature": 85.0,  
    "processor\_load": 0.9,  
    "module\_temperature": 35.0,  
    "radiation\_dose": 180.0  
}  
```
