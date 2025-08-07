import pandas as pd
import joblib
model = joblib.load("model1.pkl")
scaler = joblib.load("scaler.pkl")
df = pd.read_csv("merged_forest_files_datas_2000.csv")
df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce')
features = ['latitude', 'longitude', 'brightness', 'bright_t31', 'frp']
df = df.dropna(subset=features)
X = df[features]
X_scaled = scaler.transform(X)  
df['predicted_risk'] = model.predict(X_scaled)
df['risk_level'] = df['predicted_risk'].map({1: 'High', 0: 'Low'})
df.to_csv("fire_risk_predictions.csv", index=False)
print("fire risk predictions saved to fireriskpredictions.csv")