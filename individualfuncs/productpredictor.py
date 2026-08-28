import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

daily_ops = pd.read_csv("srp_daily_operations.csv")
wells_info = pd.read_csv("wells_master.csv")
css_cycles = pd.read_csv("css_cycle_records.csv")

merged_data = pd.merge(daily_ops, wells_info, on="well_id", how="left")
master_data = pd.merge(merged_data, css_cycles, on=["well_id", "cycle_number"], how="left")

master_data = master_data.dropna()

features = [
    'day_in_cycle',
    'steam_volume_tonnes',
    'permeability_mD',
    'net_pay_m',
    'soak_days'
]
X = master_data[features]

targets = ['oil_rate_bbl_per_day', 'reservoir_zone_temp_C']
y = master_data[targets]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

predictor_model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training the model... please wait.")
predictor_model.fit(X_train, y_train)

predictions = predictor_model.predict(X_test)

predictions_df = pd.DataFrame(predictions, columns=targets)

oil_r2 = r2_score(y_test['oil_rate_bbl_per_day'], predictions_df['oil_rate_bbl_per_day'])
temp_r2 = r2_score(y_test['reservoir_zone_temp_C'], predictions_df['reservoir_zone_temp_C'])

print(f"Model Accuracy (R-squared) for Oil Production: {oil_r2:.2f}")
print(f"Model Accuracy (R-squared) for Reservoir Temp: {temp_r2:.2f}")

new_scenario = pd.DataFrame([[45, 1200, 100, 10, 7]], columns=features)
future_prediction = predictor_model.predict(new_scenario)

print("\n--- Example Prediction ---")
print(f"Predicted Oil Rate: {future_prediction[0][0]:.1f} barrels/day")
print(f"Predicted Temperature: {future_prediction[0][1]:.1f} °C")