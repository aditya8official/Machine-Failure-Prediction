# Machine Failure Prediction

## 📌 Project Overview
This project builds a **machine learning model** to predict whether a machine is likely to fail based on sensor readings and operational parameters.  
It uses a **Random Forest Classifier** trained on historical machine performance data.  

The project includes:
- **Data preprocessing**
- **Model training**
- **Model saving/loading**
- **Prediction testing**

---

## 📂 Project Structure
```
Machine-Failure-Prediction/
├── data/
│   └── data.csv                       # Dataset containing features & failure labels
├── models/
│   ├── machine_failure_model.pkl       # Trained Random Forest model
│   └── scaler.pkl                      # StandardScaler used for preprocessing
├── scripts/
│   ├── Train.py                        # Script to train and save the model
│   └── Load.py                         # Script to load model and make predictions
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 📊 Dataset Information
- **File:** `data.csv`
- **Target Column:** `fail` (1 = failure, 0 = no failure)
- **Feature Columns:** Numerical values representing sensor readings, machine conditions, and operational data.
- **Goal:** Predict machine failure from input features.

---

## ⚙️ How to Run

### **1. Install dependencies**
```bash
pip install -r requirements.txt
```

### **2. Train the model**
```bash
python3 scripts/Train.py
```
This will:
- Train a `RandomForestClassifier`
- Save the model to `models/machine_failure_model.pkl`
- Save the scaler to `models/scaler.pkl`

### **3. Make a prediction**
```bash
python3 scripts/Load.py
```
This will load the model & scaler, then run a prediction on sample input data.

---

## 🧠 Example Output
```bash
✅ Model and scaler saved successfully.
Prediction (1 = failure, 0 = no failure): 0
```

---

## 🚀 Potential Improvements
- Add hyperparameter tuning for Random Forest.
- Integrate with a real-time sensor data API.
- Create a Streamlit web app for user-friendly predictions.
- Add feature importance visualization.

---

## 📜 License
This project can be freely used for **educational and research purposes**.
