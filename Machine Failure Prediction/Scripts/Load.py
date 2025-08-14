import joblib
import numpy as np
import pandas as pd
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "machine_failure_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"
DATA_PATH = BASE_DIR / "data" / "data.csv"  


model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
columns = pd.read_csv(DATA_PATH).drop('fail', axis=1).columns


sample_data = pd.DataFrame([[10, 3, 5, 2, 6, 1, 40, 4, 20]], columns=columns)
sample_data_scaled = scaler.transform(sample_data)
prediction = model.predict(sample_data_scaled)

print(f"Prediction (1 = failure, 0 = no failure): {prediction[0]}")