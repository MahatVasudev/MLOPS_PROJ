from flask import Flask, request, jsonify
import pandas as pd 
import joblib

app = Flask(__name__)

model = joblib.load('logistic_reg.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    df = pd.DataFrame([data])

    prediction = model.predict(df)


    return jsonify({'prediction': int(prediction[0])})



app.run(debug=True, port=5123)

"""
BODY Text
{
        "Temperature_C": 10,
        "Battery_Voltage": 10,
        "Power_Consumption_W": 10,
        "Signal_Strength_Percent": 10,
        "Fan_Speed_RPM": 10,
        "Humidity_Percent": 10,
        "Traffic_Load": 10,
        "Tower_Age_Years": 10
}
"""
