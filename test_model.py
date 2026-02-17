import joblib
import json
import pandas as pd
import xgboost as xgb

# 1. Load Model and Features
model = xgb.Booster()
model.load_model("output/models/surge_model.json")

with open("output/models/feature_names.json", "r") as f:
    feature_names = json.load(f)

# 2. Create a "Fake Scenario" (Testing)
# Farz karein: Monday subah ka time, high demand, halki baarish
test_data = pd.DataFrame([{
    "demand_60min": 150.0,
    "supply_baseline": 50.0,
    "demand_zscore": 2.5,
    "is_weekend": 0,
    "hour": 9,
    "day_of_week": 0,
    "temp": 2.0,
    "precip": 0.5,
    "is_rush_hour": 1
    # Baki features model khud handle kar lega agar zero/mean dein
}])

# Align features with training columns
for col in feature_names:
    if col not in test_data.columns:
        test_data[col] = 0

test_data = test_data[feature_names]
dtest = xgb.DMatrix(test_data)

# 3. Predict
prediction = model.predict(dtest)
print(f"\n🚀 Scenario: Monday Morning Rush + Rain")
print(f"💰 Predicted Surge Multiplier: {prediction[0]:.2f}x")