import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

print("Loading data...")
data = pd.read_csv("srp_daily_operations.csv")

data = data.dropna()

features = ['reservoir_zone_temp_C', 'oil_viscosity_cP', 'stroke_length_in', 'spm']
X = data[features]

y_oil = data['oil_rate_bbl_per_day']
y_risk = data['rod_floating_flag']

X_train, X_test, y_oil_train, y_oil_test, y_risk_train, y_risk_test = train_test_split(
    X, y_oil, y_risk, test_size=0.2, random_state=42
)

print("Training the AI brains. This might take a few seconds...")

production_ai = RandomForestRegressor(n_estimators=100, random_state=42)
production_ai.fit(X_train, y_oil_train)

risk_ai = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
risk_ai.fit(X_train, y_risk_train)

print("AI Training Complete!\n")

def find_best_pump_settings(current_temp, current_viscosity):
    spm_options = np.arange(2.0, 8.1, 0.1) 
    stroke_options = [80, 100, 120] 
    
    best_spm = 0
    best_stroke = 0
    highest_oil = 0
    
    for stroke in stroke_options:
        for spm in spm_options:
            test_scenario = pd.DataFrame([[current_temp, current_viscosity, stroke, spm]], 
                                         columns=features)
            
            risk_chance = risk_ai.predict_proba(test_scenario)[0][1] 
            predicted_oil = production_ai.predict(test_scenario)[0]
            
            if risk_chance < 0.10:
                if predicted_oil > highest_oil:
                    highest_oil = predicted_oil
                    best_spm = spm
                    best_stroke = stroke
                    
    if highest_oil == 0:
        return "Warning: The oil is too thick or cold. No safe pump settings found."
    else:
        return {
            "Best Speed (SPM)": round(best_spm, 1),
            "Best Stroke Length (inches)": best_stroke,
            "Expected Oil (barrels/day)": round(highest_oil, 1)
        }

today_temperature = 95.0
today_viscosity = 1500.0

print(f"--- TODAY'S CONDITIONS ---")
print(f"Temperature: {today_temperature} °C")
print(f"Oil Thickness: {today_viscosity} cP\n")

print("Asking the Digital Twin for advice...")
best_settings = find_best_pump_settings(today_temperature, today_viscosity)

print("\n--- DIGITAL TWIN RECOMMENDATION ---")
for key, value in best_settings.items():
    print(f"{key}: {value}")