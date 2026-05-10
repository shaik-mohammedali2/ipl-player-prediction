import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib

print("🚀 Training Started")

df = pd.read_csv("cricket_data_2026.csv")

df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(0, inplace=True)

# 🏏 FINAL 7 FEATURES (MATCH APP)
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
y = df["Wickets_Taken"]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, y)

joblib.dump(model, "ipl_model.pkl")
print("🧠 Model Saved Successfully")