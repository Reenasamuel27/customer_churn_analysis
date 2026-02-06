import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# ----------------------------
# Load dataset
# ----------------------------
import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data")

# auto-find csv file
files = os.listdir(DATA_PATH)
csv_files = [f for f in files if f.endswith(".csv")]

if not csv_files:
    raise FileNotFoundError("No CSV file found inside /data folder")

DATA_FILE = csv_files[0]
DATA_PATH = os.path.join(DATA_PATH, DATA_FILE)

print("Using dataset:", DATA_PATH)

df = pd.read_csv(DATA_PATH)
print("Dataset loaded:", df.shape)


# ----------------------------
# Auto-detect target column
# Change this if needed
# ----------------------------
TARGET = "Churn"   # <-- replace if your label column name differs

if TARGET not in df.columns:
    raise ValueError(f"Target column '{TARGET}' not found")

# ----------------------------
# Clean missing values
# ----------------------------
df = df.fillna(method="ffill")

# ----------------------------
# Encode categorical features
# ----------------------------
for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])

# ----------------------------
# Split features/target
# ----------------------------
X = df.drop(TARGET, axis=1)
y = df[TARGET]

# ----------------------------
# Train-test split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=200)
model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
pred = model.predict(X_test)
print("\nModel Performance:")
print(classification_report(y_test, pred))

# ----------------------------
# Save model
# ----------------------------
os.makedirs("../models", exist_ok=True)

joblib.dump(model, "../models/churn_model.pkl")
joblib.dump(list(X.columns), "../models/feature_columns.pkl")

print("\n✅ Model saved to /models folder")
