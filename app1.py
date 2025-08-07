
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
scaler = joblib.load('scaler.pkl')
model = joblib.load('model1.pkl')
@app.route('/')
def home():
    return render_template('home.html')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
        brightness = float(request.form['brightness'])
        bright_t31 = float(request.form['bright_t31'])
        frp = float(request.form['frp'])

        model = joblib.load('model1.pkl')
        scaler = joblib.load('scaler.pkl')

        features = np.array([[latitude, longitude, brightness, bright_t31, frp]])
        features_scaled = scaler.transform(features)
        prediction_result = model.predict(features_scaled)

        prediction = 'High Risk' if prediction_result[0] == 1 else 'Low Risk'

    return render_template('predict.html', prediction=prediction)
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        try:
            file = request.files['file']
            df = pd.read_csv(file)
            required_columns = ['latitude', 'longitude', 'brightness', 'bright_t31', 'frp']
            if not all(col in df.columns for col in required_columns):
                return "CSV must contain columns: " + ", ".join(required_columns)
            X = df[required_columns]
            X_scaled = scaler.transform(X)
            df['risk_level'] = model.predict(X_scaled)
            df['risk_level'] = df['risk_level'].map({0: 'Low', 1: 'High'})
            df['risk_score'] = model.predict_proba(X_scaled)[:, 1]
            output_path = 'static/predicted_output.csv'
            df.to_csv(output_path, index=False)

            return render_template('upload.html', 
                                   csv_result="Prediction completed. <a href='static/predicted_output.csv'>Download CSV</a>", 
                                   table=df.head().to_html(classes='data', header="true")) 
        except Exception as e:
            return f"Error: {e}"
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
