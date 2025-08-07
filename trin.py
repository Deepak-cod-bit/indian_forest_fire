from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd 
df = pd.read_csv("merged_forest_files_datas_2000.csv")
print("Unique values in 'confidence':", df['confidence'].unique())
if df['confidence'].dtype != 'object':
    df = df[df['confidence'].notna()]  
    df['confidence_label'] = df['confidence'].apply(lambda x: 1 if x >= 80 else 0)
else:
    df['confidence'] = df['confidence'].astype(str).str.lower()
    df = df[df['confidence'].isin(['low', 'high'])]
    df['confidence']
print(f"Rows after filtering: {len(df)}")
features = ['latitude', 'longitude', 'brightness', 'bright_t31', 'frp']
X = df[features]
y = df['confidence_label']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
joblib.dump(model, 'model1.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Training completed and model saved.")
