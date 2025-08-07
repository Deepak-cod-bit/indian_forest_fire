# 🌲 Forest Fire Prediction using Machine Learning 🔥

This project predicts the risk of forest fires based on environmental data using a machine learning model. It helps in identifying high-risk areas to support early prevention efforts.

## 🧠 Project Highlights
- **ML Model**: Logistic Regression
- **Features Used**: Latitude, Longitude, Brightness, Bright_T31, FRP
- **Risk Classification**: Uses a confidence threshold to define "High Risk"
- **Web Interface**: Built using Flask + HTML
- **Prediction Modes**:
  - Single Prediction (form input)
  - Batch Prediction (CSV upload)

## 📁 Project Structure
| File/Folder      | Description                                |
|------------------|--------------------------------------------|
| `trin.py`        | Training script for model & scaler         |
| `model1.pkl`     | Trained ML model (saved using joblib)      |
| `scaler.pkl`     | Scaler used for normalization              |
| `predict.py`     | Backend logic for handling predictions     |
| `base.html`      | Base HTML template                         |
| `home.html`      | Homepage with app description              |
| `predict.html`   | Form-based prediction UI                   |
| `upload.html`    | File upload page for batch prediction      |

## 🔧 Libraries & Tools Used
- Python, Pandas, NumPy, Scikit-learn
- Flask (Web App Framework)
- HTML/CSS (Frontend Templates)
- Joblib (Model Serialization)

## 🚀 How to Run Locally
1. Clone the repository  
2. Run `pip install -r requirements.txt` *(you can generate this file using `pip freeze`)*  
3. Start the Flask app:
   ```bash
   python predict.py
