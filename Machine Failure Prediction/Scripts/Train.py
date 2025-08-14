import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
MODEL_PATH = BASE_DIR / "models" / "machine_failure_model.pkl"
SCALER_PATH = BASE_DIR / "models" / "scaler.pkl"


data = pd.read_csv(DATA_PATH)


X = data.drop('fail', axis=1)
y = data['fail']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = RandomForestClassifier(random_state=42)
model.fit(X_scaled, y)


MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)  
joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("✅ Model and scaler saved successfully.")