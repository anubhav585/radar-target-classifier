import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import extract_statistical_features
from src.models import get_random_forest_model 

app = Flask(__name__)

model = None
scaler = None
class_names = ["Bird", "Drone", "Aircraft", "Stealth UAV"]

def initialize_model():
    global model, scaler
    print("Loading data and training modular Random Forest model...")
    df = pd.read_csv("data/raw/astra_dataset.csv")
    feature_cols = [str(i) for i in range(300)]
    
    X_tab = extract_statistical_features(df, feature_cols)
    y = df['label'].values
    
    X_train, _, y_train, _ = train_test_split(X_tab, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Using the exact same function from src/models.py
    model = get_random_forest_model(n_estimators=100)
    model.fit(X_train_scaled, y_train)
    print("Model initialized successfully!")

initialize_model()

@app.route('/', methods=['GET'])
def home():
    """Root endpoint to verify the API is running."""
    return jsonify({
        "status": "online",
        "message": "Radar Target Classification API is running.",
        "usage": "Send a POST request to /predict with JSON: {'features': [300 floats]}"
    })

@app.route('/predict', methods=['POST'])
def predict_target():
    data = request.get_json()
    
    if not data or 'features' not in data:
        return jsonify({"error": "Missing 'features' list in request JSON."}), 400
        
    features = data['features']
    if len(features) != 300:
        return jsonify({"error": "Input must contain exactly 300 features."}), 400
    
    # Use modular feature extraction
    df_sample = pd.DataFrame([features], columns=[str(i) for i in range(300)])
    tab_features = extract_statistical_features(df_sample, [str(i) for i in range(300)])
    
    scaled_features = scaler.transform(tab_features)
    pred_idx = model.predict(scaled_features)[0]
    probabilities = model.predict_proba(scaled_features)[0]
    
    return jsonify({
        "predicted_class": class_names[pred_idx],
        "confidence": float(probabilities[pred_idx]),
        "class_probabilities": {class_names[i]: float(probabilities[i]) for i in range(4)}
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)