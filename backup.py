import streamlit as st
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

st.set_page_config(page_title="Digital Twin Control Room", layout="wide", initial_sidebar_state="collapsed")

@st.cache_resource
def load_and_train_models():
    daily_ops = pd.read_csv("srp_daily_operations.csv").dropna()
    
    features = ['reservoir_zone_temp_C', 'oil_viscosity_cP', 'stroke_length_in', 'spm']
    X = daily_ops[features]
    
    pump_ai = RandomForestRegressor(n_estimators=100, random_state=42)
    pump_ai.fit(X, daily_ops['oil_rate_bbl_per_day'])

    health_ai = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    health_ai.fit(X, daily_ops['rod_floating_flag'])
    
    return pump_ai, health_ai

pump_ai, health_ai = load_and_train_models()

def get_optimal_settings(temp, visc):
    best_spm, best_stroke, max_oil, risk_level = 2.0, 80, 0, 0
    
    for stroke in [80, 100, 120]:
        for spm in np.arange(2.0, 8.0, 0.5):
            test_df = pd.DataFrame([[temp, visc, stroke, spm]], 
                                   columns=['reservoir_zone_temp_C', 'oil_viscosity_cP', 'stroke_length_in', 'spm'])
            
            risk_prob = health_ai.predict_proba(test_df)[0][1]
            oil_pred = pump_ai.predict(test_df)[0]
            
            if risk_prob < 0.25 and oil_pred > max_oil:
                max_oil = oil_pred
                best_spm = spm
                best_stroke = stroke
                risk_level = risk_prob
                
    if max_oil == 0:
        test_df = pd.DataFrame([[temp, visc, 80, 2.0]], 
                               columns=['reservoir_zone_temp_C', 'oil_viscosity_cP', 'stroke_length_in', 'spm'])
        max_oil = pump_ai.predict(test_df)[0]
        risk_level = health_ai.predict_proba(test_df)[0][1]
        
    return best_stroke, best_spm, max_oil, risk_level

st.title("🛢️ Well-to-Surface Digital Twin")
st.markdown("### Advanced Control Room")

top_metrics = st.empty()
middle_panels = st.empty()
bottom_charts = st.empty()

history_time, history_oil, history_temp, history_load = [], [], [], []

current_temp = 140.0
current_viscosity = 800.0
day_counter = 1

while True:
    temp_drop = np.random.uniform(1.0, 2.0)
    current_temp -= temp_drop
    
    visc_rise = np.random.uniform(20.0, 40.0)
    current_viscosity += visc_rise
    day_counter += 1
    
    opt_stroke, opt_spm, expected_oil, risk_prob = get_optimal_settings(current_temp, current_viscosity)
    
    motor_load = 5.0 + (current_viscosity / 300) + (opt_spm * 0.5)
    pump_fillage = max(40, 95 - (current_viscosity / 50)) 
    
    temp_threshold = 60.0
    days_to_steam = max(0, int((current_temp - temp_threshold) / 1.5)) 
    
    history_time.append(day_counter)
    history_oil.append(expected_oil)
    history_temp.append(current_temp)
    history_load.append(motor_load)
    
    if len(history_time) > 30:
        history_time.pop(0)
        history_oil.pop(0)
        history_temp.pop(0)
        history_load.pop(0)

    with top_metrics.container():
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Reservoir Temp", f"{current_temp:.1f} °C", f"-{temp_drop:.1f} °C")
        c2.metric("Oil Viscosity", f"{current_viscosity:.0f} cP", f"+{visc_rise:.0f} cP", delta_color="inverse")
        c3.metric("Pump Fillage", f"{pump_fillage:.0f} %")
        c4.metric("Motor Load", f"{motor_load:.1f} kW")
        c5.metric("Target Stroke", f"{opt_stroke} in")
        c6.metric("Target Speed", f"{opt_spm} SPM")
        st.divider()

    with middle_panels.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("🛠️ System Health")
            if risk_prob > 0.50:
                st.error("🚨 CRITICAL ALARM: High impact loading risk. Speed reduced.")
            elif risk_prob > 0.25:
                st.warning("⚠️ WARNING: Equipment stress rising. Monitoring.")
            else:
                st.success("✅ SYSTEM HEALTHY: Zero rod floating risk detected.")
                
        with col2:
            st.subheader("🔥 Steam Planner")
            if days_to_steam > 15:
                st.info(f"⏳ Next Steam Injection in **~{days_to_steam} days**.")
            elif days_to_steam > 0:
                st.warning(f"⏰ PREPARE BOILERS: Injection in **{days_to_steam} days**.")
            else:
                st.error("🛑 INITIATE STEAM INJECTION NOW.")
                
        with col3:
            fig_gauge = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = risk_prob * 100,
                title = {'text': "Rod Failure Risk (%)"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "white"},
                    'steps': [
                        {'range': [0, 25], 'color': "green"},
                        {'range': [25, 50], 'color': "orange"},
                        {'range': [50, 100], 'color': "red"}],
                }))
            fig_gauge.update_layout(height=200, margin=dict(l=20, r=20, t=30, b=20))
            
            st.plotly_chart(fig_gauge, use_container_width=True, key=f"gauge_{day_counter}")

    with bottom_charts.container():
        chart_col1, chart_col2 = st.columns([2, 1])
        
        with chart_col1:
            fig_main = go.Figure()
            
            fig_main.add_trace(go.Scatter(
                x=history_time, y=history_oil, 
                name="Oil Flow (bbl/day)", 
                mode='lines',
                fill='tozeroy', 
                line=dict(color='#00FF00', width=2)
            ))
            
            fig_main.add_trace(go.Scatter(
                x=history_time, y=history_temp, 
                name="Temperature (°C)", 
                mode='lines',
                line=dict(color='#FF4B4B', width=3, dash='dot'),
                yaxis="y2"
            ))
            
            fig_main.update_layout(
                title="Production & Thermal Decline Curve",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                yaxis=dict(title=dict(text="Oil Flow", font=dict(color="#00FF00")), tickfont=dict(color="#00FF00")),
                yaxis2=dict(title=dict(text="Temp (°C)", font=dict(color="#FF4B4B")), tickfont=dict(color="#FF4B4B"), overlaying="y", side="right"),
                height=350, margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_main, use_container_width=True, key=f"main_chart_{day_counter}")
            
        with chart_col2:
            fig_bar = go.Figure(go.Bar(
                x=history_time[-10:], 
                y=history_load[-10:],
                marker_color='#1E90FF'
            ))
            fig_bar.update_layout(
                title="Motor Load (Last 10 Days)",
                yaxis_title="Load (kW)",
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(l=0, r=0, t=40, b=0)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True, key=f"bar_chart_{day_counter}")

    time.sleep(2.5)