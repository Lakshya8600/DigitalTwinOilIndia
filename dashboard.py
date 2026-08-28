import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

st.set_page_config(page_title="Well Operations Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap');
:root { --ink:#17212b; --muted:#667580; --line:#dce4e7; --teal:#087f8c; --orange:#e87531; --mint:#e8f5f1; }
html, body, [class*="css"] { font-family:'Space Grotesk',sans-serif; color:var(--ink); }
.stApp { background:linear-gradient(135deg,#f5f8f7 0%,#fff 48%,#eef5f4 100%); }
section[data-testid="stMain"] * { color:var(--ink); }
section[data-testid="stMain"] .section-label, section[data-testid="stMain"] .eyebrow { color:var(--teal); }
section[data-testid="stMain"] .subtitle { color:var(--muted); }
[data-testid="stSidebar"] { background:#152932; border-right:0; }
[data-testid="stSidebar"] * { color:#eaf3f1 !important; }
[data-testid="stSidebar"] .stCaption { color:#a9c1c1 !important; }
h1,h2,h3 { letter-spacing:0; } h1 { font-size:2.35rem !important; font-weight:700; margin-bottom:0 !important; }
.eyebrow,.section-label { color:var(--teal); font:500 .72rem 'DM Mono',monospace; letter-spacing:.1em; text-transform:uppercase; }
.subtitle { color:var(--muted); margin:.1rem 0 1.4rem; } .status { display:inline-block; padding:.35rem .7rem; border-radius:4px; background:#dff3ec; color:#087256; font:500 .72rem 'DM Mono',monospace; }
.section-label { color:#5e7079; margin:1.1rem 0 .55rem; } .callout { border-left:4px solid var(--teal); background:var(--mint); padding:.8rem 1rem; border-radius:0 6px 6px 0; }
div[data-testid="stMetric"] { background:rgba(255,255,255,.75); border:1px solid var(--line); border-radius:8px; padding:10px 14px; }
div[data-testid="stMetricLabel"] p { font-family:'DM Mono',monospace; font-size:.7rem; } .stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    daily = pd.read_csv("srp_daily_operations.csv", parse_dates=["date"])
    cycles = pd.read_csv("css_cycle_records.csv", parse_dates=["injection_start_date", "production_start_date"])
    failures = pd.read_csv("rod_failure_log.csv", parse_dates=["date"])
    wells = pd.read_csv("wells_master.csv")
    return daily.dropna(), cycles.dropna(), failures.dropna(), wells.dropna()


@st.cache_resource
def train_models(daily):
    features = ["reservoir_zone_temp_C", "oil_viscosity_cP", "stroke_length_in", "spm"]
    model_input = daily[features]
    production = RandomForestRegressor(n_estimators=160, random_state=42).fit(model_input, daily["oil_rate_bbl_per_day"])
    health = RandomForestClassifier(n_estimators=160, class_weight="balanced", random_state=42).fit(model_input, daily["rod_floating_flag"])
    return production, health, features


def risk_for(model, scenario):
    probabilities = model.predict_proba(scenario)[0]
    classes = list(model.classes_)
    return float(probabilities[classes.index(True)]) if True in classes else 0.0


def recommendation(production, health, features, temp, viscosity, risk_limit):
    options = []
    for stroke in [80, 100, 120]:
        for spm in np.arange(2.0, 8.1, 0.2):
            scenario = pd.DataFrame([[temp, viscosity, stroke, spm]], columns=features)
            options.append({"stroke": stroke, "spm": round(float(spm), 1), "oil": float(production.predict(scenario)[0]), "risk": risk_for(health, scenario)})
    candidates = pd.DataFrame(options)
    safe = candidates[candidates["risk"] <= risk_limit]
    chosen = (safe if not safe.empty else candidates).sort_values("oil", ascending=False).iloc[0]
    candidates["safe"] = candidates["risk"] <= risk_limit
    return chosen, candidates


def chart_layout(height=340):
    return dict(height=height, margin=dict(l=8, r=8, t=42, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Space Grotesk", color="#52636b"), legend=dict(orientation="h", y=1.08, x=0))


daily, cycles, failures, wells = load_data()
production, health, features = train_models(daily)

well_ids = sorted(wells["well_id"].unique())
with st.sidebar:
    st.markdown("## ◈ WELLSIGHT")
    st.caption("OPERATIONS INTELLIGENCE / v2.0")
    st.divider()
    selected_well = st.selectbox("Asset focus", well_ids, index=0)
    well_daily = daily[daily["well_id"] == selected_well].sort_values("date")
    latest = well_daily.iloc[-1]
    st.markdown("#### Scenario controls")
    temperature = st.slider("Reservoir-zone temperature (°C)", 40.0, 200.0, float(latest["reservoir_zone_temp_C"]), .5)
    viscosity = st.slider("Oil viscosity (cP)", 50.0, 4000.0, float(latest["oil_viscosity_cP"]), 10.0)
    risk_limit_pct = st.slider("Maximum accepted rod risk", 5, 60, 25, 1, format="%d%%")
    risk_limit = risk_limit_pct / 100
    lookback = st.slider("Trend window (days)", 15, min(120, len(well_daily)), min(60, len(well_daily)))
    st.divider()
    st.caption("Model inputs are editable scenario values. Historical charts remain grounded in recorded operating data.")

chosen, options = recommendation(production, health, features, temperature, viscosity, risk_limit)
predicted_fill = max(0, min(100, 98 - viscosity / 65 - chosen["spm"] * 1.2))
predicted_load = 5 + viscosity / 300 + chosen["spm"] * .5
current_risk = float(chosen["risk"])
asset_cycles = cycles[cycles["well_id"] == selected_well]
asset_failures = failures[failures["well_id"] == selected_well]
asset_info = wells[wells["well_id"] == selected_well].iloc[0]

st.markdown('<div class="eyebrow">FIELD COMMAND / DIGITAL TWIN</div>', unsafe_allow_html=True)
st.title("Well Operations Intelligence")
st.markdown(f'<div class="subtitle">Decision cockpit for <b>{selected_well}</b> · last telemetry {latest["date"].strftime("%d %b %Y")} <span class="status">● MODEL ONLINE</span></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Live operating picture</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Oil rate", f"{latest['oil_rate_bbl_per_day']:.1f} bbl/d", f"{latest['water_cut_frac'] * 100:.1f}% water cut", delta_color="off")
m2.metric("Scenario output", f"{chosen['oil']:.1f} bbl/d", f"{chosen['oil'] - latest['oil_rate_bbl_per_day']:+.1f} vs latest")
m3.metric("Rod risk", f"{current_risk * 100:.1f}%", f"limit {risk_limit * 100:.0f}%", delta_color="inverse")
m4.metric("Fillage", f"{predicted_fill:.1f}%", f"{latest['pump_fillage_frac'] * 100:.1f}% observed")
m5.metric("Motor load", f"{predicted_load:.1f} kW", f"{latest['motor_load_kW']:.1f} kW observed")
m6.metric("Recommended", f"{chosen['stroke']:.0f} in / {chosen['spm']:.1f} SPM", "optimized setpoint")

left, center, right = st.columns([1.1, 1.4, 1.1])
with left:
    st.markdown('<div class="section-label">Recommended action</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="callout"><b>Operate at {chosen["stroke"]:.0f} in and {chosen["spm"]:.1f} SPM.</b><br>Projected {chosen["oil"]:.1f} bbl/d while keeping rod risk at {current_risk * 100:.1f}%.</div>', unsafe_allow_html=True)
    st.write("")
    st.write(f"**Scenario delta:** {temperature - latest['reservoir_zone_temp_C']:+.1f} °C and {viscosity - latest['oil_viscosity_cP']:+.0f} cP from latest telemetry.")
with center:
    gauge = go.Figure(go.Indicator(mode="gauge+number", value=current_risk * 100, number={"suffix": "%"}, title={"text": "ROD FLOATING RISK", "font": {"size": 13}}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#e87531"}, "steps": [{"range": [0, 25], "color": "#dff3ec"}, {"range": [25, 50], "color": "#fff0dc"}, {"range": [50, 100], "color": "#f8dddd"}], "threshold": {"line": {"color": "#152932", "width": 3}, "value": risk_limit * 100}}))
    gauge.update_layout(**chart_layout(220))
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})
with right:
    st.markdown('<div class="section-label">Asset profile</div>', unsafe_allow_html=True)
    profile = pd.DataFrame({"Parameter": ["Depth", "API gravity", "Permeability", "Net pay", "Completion", "Rod grade"], "Value": [f"{asset_info['depth_m']:.0f} m", f"{asset_info['api_gravity']:.2f}°", f"{asset_info['permeability_mD']:.0f} mD", f"{asset_info['net_pay_m']:.1f} m", str(int(asset_info['completion_year'])), asset_info['rod_string_grade']]})
    st.dataframe(profile, hide_index=True, use_container_width=True, height=245)

st.markdown('<div class="section-label">Performance intelligence</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["Production & thermal", "Operating envelope", "Reliability & cycles"])
with tab1:
    trend = well_daily.tail(lookback)
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(go.Scatter(x=trend["date"], y=trend["oil_rate_bbl_per_day"], name="Oil rate", line={"color": "#087f8c", "width": 3}, fill="tozeroy", fillcolor="rgba(8,127,140,.12)"), secondary_y=False)
    figure.add_trace(go.Scatter(x=trend["date"], y=trend["reservoir_zone_temp_C"], name="Temperature", line={"color": "#e87531", "width": 2}), secondary_y=True)
    figure.add_trace(go.Scatter(x=trend["date"], y=trend["oil_viscosity_cP"], name="Viscosity", line={"color": "#506b83", "width": 1.5, "dash": "dot"}), secondary_y=True)
    figure.update_yaxes(title_text="Oil rate (bbl/d)", secondary_y=False)
    figure.update_yaxes(title_text="Thermal state / viscosity", secondary_y=True)
    figure.update_layout(title="Recorded decline and fluid behavior", **chart_layout(390))
    st.plotly_chart(figure, use_container_width=True)
    a, b, c = st.columns(3)
    a.metric("Avg. production", f"{trend['oil_rate_bbl_per_day'].mean():.1f} bbl/d")
    b.metric("Avg. efficiency", f"{trend['pump_volumetric_efficiency'].mean() * 100:.1f}%")
    c.metric("Avg. polished-rod load", f"{trend['polished_rod_load_lbs'].mean():,.0f} lb")
with tab2:
    envelope = options.copy()
    envelope["risk_pct"] = envelope["risk"] * 100
    heat = go.Figure(go.Scatter(x=envelope["spm"], y=envelope["oil"], mode="markers", marker={"size": 9, "color": envelope["risk_pct"], "colorscale": [[0, "#087f8c"], [.25, "#e8bb57"], [1, "#d34e42"]], "colorbar": {"title": "Risk %"}, "line": {"width": .5, "color": "white"}}, text=[f"{s:.0f} in / {v:.1f} SPM" for s, v in zip(envelope["stroke"], envelope["spm"])], hovertemplate="%{text}<br>Oil: %{y:.1f} bbl/d<extra></extra>"))
    heat.add_trace(go.Scatter(x=[chosen["spm"]], y=[chosen["oil"]], mode="markers", marker={"size": 16, "color": "#17212b", "symbol": "star"}, name="Recommended"))
    heat.update_layout(title="AI-tested pump operating envelope", xaxis_title="Speed (SPM)", yaxis_title="Predicted oil rate (bbl/d)", **chart_layout(390))
    st.plotly_chart(heat, use_container_width=True)
    st.caption(f"{len(envelope[envelope['safe']])} of {len(envelope)} tested settings satisfy the {risk_limit * 100:.0f}% risk limit.")
with tab3:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        cycle = asset_cycles.sort_values("cycle_number")
        cycle_fig = go.Figure()
        cycle_fig.add_trace(go.Bar(x=cycle["cycle_number"], y=cycle["cycle_oil_produced_bbl"], name="Cycle oil", marker_color="#087f8c"))
        cycle_fig.add_trace(go.Scatter(x=cycle["cycle_number"], y=cycle["steam_oil_ratio_SOR"] * cycle["cycle_oil_produced_bbl"].max() / 2, name="SOR (scaled)", line={"color": "#e87531", "width": 3}, mode="lines+markers"))
        cycle_fig.update_layout(title="CSS cycle yield and steam efficiency", xaxis_title="Cycle", yaxis_title="Oil produced (bbl)", **chart_layout(360))
        st.plotly_chart(cycle_fig, use_container_width=True)
    with c2:
        mode_counts = asset_failures["failure_mode"].value_counts()
        failure_fig = go.Figure(go.Bar(x=mode_counts.values, y=mode_counts.index, orientation="h", marker_color="#d34e42"))
        failure_fig.update_layout(title=f"Failure history · {len(asset_failures)} events", xaxis_title="Events", **chart_layout(360))
        st.plotly_chart(failure_fig, use_container_width=True)

st.markdown('<div class="section-label">Signal register</div>', unsafe_allow_html=True)
signal_columns = ["date", "day_in_cycle", "oil_rate_bbl_per_day", "water_cut_frac", "fluid_level_pct", "pump_volumetric_efficiency", "polished_rod_load_lbs", "rod_floating_risk_score"]
signals = well_daily.tail(12)[signal_columns].copy()
signals.columns = ["Date", "Cycle day", "Oil bbl/d", "Water cut", "Fluid level %", "Vol. efficiency", "Rod load lb", "Risk score"]
signals["Date"] = signals["Date"].dt.strftime("%d %b %Y")
st.dataframe(signals.sort_values("Date", ascending=False), hide_index=True, use_container_width=True, column_config={"Water cut": st.column_config.NumberColumn(format="%.1f%%"), "Vol. efficiency": st.column_config.NumberColumn(format="%.1f%%"), "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})