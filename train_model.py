'''import pandas as pd

# Load Dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

print("===== First 5 Rows =====")
print(df.head())

print("\n===== Shape of Dataset =====")
print(df.shape)

print("\n===== Dataset Information =====")
print(df.info())

print("\n===== Missing Values =====")
print(df.isnull().sum())

print("\n===== Crop Classes =====")
print(df['label'].unique())'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")

# Features (Input)
X = df.drop("label", axis=1)

# Target (Output)
y = df["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Create model
model = RandomForestClassifier(random_state=42)

# Train model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")
joblib.dump(model, "model/crop_model.pkl")

print("Model saved successfully!")