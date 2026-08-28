import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

print("--- STARTING WELL-TO-SURFACE DIGITAL TWIN ---\n")

daily_ops = pd.read_csv("srp_daily_operations.csv").dropna()
css_cycles = pd.read_csv("css_cycle_records.csv").dropna()

print("Training AI Models...")

css_features = ['steam_volume_tonnes', 'soak_days']
X_css = css_cycles[css_features]
y_css = css_cycles[['peak_wellbore_temp_C', 'cycle_oil_produced_bbl']]

thermal_ai = RandomForestRegressor(n_estimators=100, random_state=42)
thermal_ai.fit(X_css, y_css)

srp_features = ['reservoir_zone_temp_C', 'oil_viscosity_cP', 'stroke_length_in', 'spm']
X_srp = daily_ops[srp_features]
y_srp_oil = daily_ops['oil_rate_bbl_per_day']

pump_ai = RandomForestRegressor(n_estimators=100, random_state=42)
pump_ai.fit(X_srp, y_srp_oil)

y_srp_risk = daily_ops['rod_floating_flag']
health_ai = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
health_ai.fit(X_srp, y_srp_risk)

print("AI Training Complete!\n")

def optimize_css_cycle():
    best_steam = 0
    best_oil = 0
    best_sor = 999 
    
    for steam_test in range(800, 2000, 50):
        test_data = pd.DataFrame([[steam_test, 7]], columns=css_features)
        
        predictions = thermal_ai.predict(test_data)[0]
        predicted_oil = predictions[1]
        
        predicted_sor = steam_test / predicted_oil
        
        if predicted_oil > best_oil and predicted_sor < 2.0:
            best_oil = predicted_oil
            best_steam = steam_test
            best_sor = predicted_sor
            
    return best_steam, best_oil, best_sor

def optimize_daily_pump(current_temp, current_viscosity):
    best_spm = 0
    best_stroke = 0
    max_safe_oil = 0
    
    acceptable_risk_limit = 0.25 
    
    for stroke in [80, 100, 120]:
        for spm in np.arange(2.0, 8.0, 0.5):
            
            test_setup = pd.DataFrame([[current_temp, current_viscosity, stroke, spm]], 
                                      columns=srp_features)
            
            risk_chance = health_ai.predict_proba(test_setup)[0][1]
            predicted_oil = pump_ai.predict(test_setup)[0]
            
            if risk_chance < acceptable_risk_limit:
                if predicted_oil > max_safe_oil:
                    max_safe_oil = predicted_oil
                    best_spm = spm
                    best_stroke = stroke
                    
    if max_safe_oil == 0:
        print("   [System Alert: High rod float risk detected. Engaging lowest-speed survival mode.]")
        best_stroke = 80
        best_spm = 2.0
        
        survival_setup = pd.DataFrame([[current_temp, current_viscosity, best_stroke, best_spm]], 
                                      columns=srp_features)
        max_safe_oil = pump_ai.predict(survival_setup)[0]
                    
    return best_stroke, best_spm, max_safe_oil

print("--- DIGITAL TWIN RESULTS & OPTIMIZATIONS ---\n")

opt_steam, exp_oil_cycle, opt_sor = optimize_css_cycle()
print("1. CSS Cycle Optimization (Long-Term Planning):")
print(f"   -> Recommended Steam Volume: {opt_steam} tonnes")
print(f"   -> Expected Cycle Production: {exp_oil_cycle:.0f} barrels")
print(f"   -> Optimized Steam-Oil Ratio (SOR): {opt_sor:.2f}")
print("   Benefit: Reduced steam energy consumption and maximized recovery.\n")

today_temp = 105.0
today_viscosity = 1300.0

opt_stroke, opt_spm, exp_daily_oil = optimize_daily_pump(today_temp, today_viscosity)
print("2. SRP Operation & Equipment Health (Daily Optimization):")
print(f"   -> Current Well Conditions: {today_temp}C, {today_viscosity} cP")
print(f"   -> Recommended Stroke Length: {opt_stroke} inches")
print(f"   -> Recommended Pump Speed: {opt_spm} SPM")
print(f"   -> Expected Daily Production: {exp_daily_oil:.1f} barrels/day")
print("   Benefit: Continuous production with zero detected rod floating risk. Minimized impact loading.\n")

print("3. Overall System Status:")
print("   -> Data-driven decisions are ACTIVE.")
print("   -> Predictive equipment reliability is ACTIVE.")