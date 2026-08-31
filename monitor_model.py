import pandas as pd 
import joblib
import mlflow

from sklearn.metrics import accuracy_score

model = joblib.load("logistic_reg.joblib")

print('Model loaded successfully')

new_data = pd.read_excel('new_tower_telemetry.xlsx') # I dont have this file 😭

features = [
        "Temperature_C",
        "Battery_Voltage",
        "Power_Consumption_W",
        "Signal_Strength_Percent",
        "Fan_Speed_RPM",
        "Humidity_Percent",
        "Traffic_Load",
        "Tower_Age_Years"
]

X_new = new_data[features]

y_new = new_data["Failure_Within_48Hrs"]

predictions = model.predict(X_new)

accuracy = accuracy_score(y_new, predictions)

print('Production Accuracy', round(accuracy, 2))

mlflow.set_experiment('Telecom_Tower_Model_Monitoring')

with mlflow.start_run():
    mlflow.log_param('model', 'logistic regression')

    mlflow.log_param('n_iter', 10000)

    mlflow.log_metric('accuracy', accuracy)


    threshold = 0.8

    if accuracy < threshold:
        print()
        print('Warning: Model performance has degraded')

    else:
        print()
        print('Model performance is stable')
