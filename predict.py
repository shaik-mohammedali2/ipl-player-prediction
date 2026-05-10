import numpy as np
import pandas as pd
import joblib

print("🚀 Prediction Started")

# Load model
model = joblib.load("ipl_model.pkl")

# Load dataset
df = pd.read_csv("cricket_data_2026.csv")
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(0, inplace=True)

# EXACT SAME FEATURES AS MODEL
features = [
    "Matches_Batted",
    "Runs_Scored",
    "Balls_Faced",
    "Fours",
    "Sixes",
    "Economy_Rate",
    "Bowling_Strike_Rate"
]

X = df[features]

# use safe sample row
input_data = pd.DataFrame([X.iloc[0].values], columns=features)

# prediction
prediction = model.predict(input_data)
print("🎯 Predicted Wickets:", prediction[0])