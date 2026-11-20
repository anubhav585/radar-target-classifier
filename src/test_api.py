import requests
import json
import pandas as pd

# Load a real sample from your dataset (e.g., first row)
df = pd.read_csv("data/raw/astra_dataset.csv")
feature_cols = [str(i) for i in range(300)]
real_features = df.iloc[0][feature_cols].tolist()

# Cast label to integer to fix indexing
true_label_idx = int(df.iloc[0]['label'])

class_names = ["Bird", "Drone", "Aircraft", "Stealth UAV"]
print(f"Testing API with a real sample. True Label: {class_names[true_label_idx]}")

payload = {"features": real_features}
headers = {"Content-Type": "application/json"}

response = requests.post("http://127.0.0.1:5000/predict", data=json.dumps(payload), headers=headers)

print("Status Code:", response.status_code)
print("Response JSON:", response.json())