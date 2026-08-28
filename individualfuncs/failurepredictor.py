import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

daily_ops = pd.read_csv("srp_daily_operations.csv")

data = daily_ops.dropna()


features = [
    'oil_viscosity_cP',           
    'motor_load_kW',              
    'reservoir_zone_temp_C',      
    'spm',                        
    'polished_rod_load_lbs'       
]
X = data[features]


targets = ['rod_floating_flag', 'rod_failure_event']
y = data[targets]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


classifier_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)

print("Training the Failure Predictor... please wait.")
classifier_model.fit(X_train, y_train)


predictions = classifier_model.predict(X_test)


predictions_df = pd.DataFrame(predictions, columns=targets)


print("\n--- Accuracy Report for Rod Floating ---")
print(classification_report(y_test['rod_floating_flag'], predictions_df['rod_floating_flag']))

print("\n--- Accuracy Report for Rod Failure ---")
print(classification_report(y_test['rod_failure_event'], predictions_df['rod_failure_event']))


current_conditions = pd.DataFrame([[1600.0, 9.5, 95.0, 5.0, 13500.0]], columns=features)
warning_check = classifier_model.predict(current_conditions)

print("\n--- Real-Time System Check ---")
print(f"Rod Floating Risk Detected: {'YES WARNING' if warning_check[0][0] else 'Normal'}")
print(f"Rod Failure Risk Detected: {'YES WARNING' if warning_check[0][1] else 'Normal'}")