import pandas as pd
import numpy as np

def optimize_pump_settings(current_temp, current_viscosity, production_ai, risk_ai):
    """
    Finds the best SPM and Stroke Length to maximize oil without breaking the rod.
    
    Inputs:
    - current_temp: Today's well temperature
    - current_viscosity: Today's oil thickness
    - production_ai: Your trained model that predicts oil flow
    - risk_ai: Your trained model that predicts rod floating risk
    """
    
    spm_options = np.arange(2.0, 8.1, 0.1) 
    
    stroke_options = [80, 100, 120] 
    
    best_spm = 0
    best_stroke = 0
    highest_oil_production = 0
    
    # 2. Test every single combination (thousands of loops)
    for stroke in stroke_options:
        for spm in spm_options:
            
            # Create a test scenario for the AI to look at
            test_data = pd.DataFrame([[current_temp, current_viscosity, stroke, spm]], 
                                     columns=['reservoir_zone_temp_C', 'oil_viscosity_cP', 
                                              'stroke_length_in', 'spm'])
            
            predicted_risk = risk_ai.predict(test_data)[0]
            predicted_oil = production_ai.predict(test_data)[0]
            
            
            if predicted_risk < 0.1:
                
               
                if predicted_oil > highest_oil_production:
                    highest_oil_production = predicted_oil
                    best_spm = spm
                    best_stroke = stroke
                    
   
    if highest_oil_production == 0:
        return "Warning: No safe pump settings found. Shut down or inject steam."
    else:
        return {
            "Recommended SPM": round(best_spm, 1),
            "Recommended Stroke Length": best_stroke,
            "Expected Oil (bbl/day)": round(highest_oil_production, 1)
        }



best_settings = optimize_pump_settings(95.0, 1200.0, predictor_model, classifier_model)
print(best_settings)  