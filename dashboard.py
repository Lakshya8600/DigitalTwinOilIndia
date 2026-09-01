import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


st.set_page_config(page_title="Well Operations Intelligence", page_icon="◈", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1117;
    --panel: #121a22;
    --panel-hover: #17222c;
    --line: #263541;

    --text: #edf2f5;
    --muted: #8c9aa5;

    --accent: #e0b83f;
    --cyan: #42c6d4;
    --green: #49b88a;
    --orange: #e39a52;
    --red: #d86658;
}

/* ---------- GLOBAL ---------- */

html, body, [class*="st-"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 85% 5%, rgba(66,198,212,.06), transparent 25%),
        var(--bg);
}

section[data-testid="stMain"] {
    color: var(--text);
}

section[data-testid="stMain"] .block-container {
    max-width: 1450px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

/* ---------- TOP HEADER ---------- */

header[data-testid="stHeader"] {
    background: rgba(11,17,23,.94);
    border-bottom: 1px solid var(--line);
}

/* ---------- SIDEBAR ---------- */

[data-testid="stSidebar"] {
    background: #090f14;
    border-right: 1px solid var(--line);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: var(--muted) !important;
}

[data-testid="stSidebar"] hr {
    border-color: var(--line);
}

[data-testid="stSidebar"] label {
    color: #cbd5db !important;
    font-size: .82rem !important;
}

/* ---------- BRAND ---------- */

.brand {
    font: 600 .82rem 'IBM Plex Mono', monospace;
    letter-spacing: .2em;
    color: var(--accent) !important;
    margin: 0;
}

.brand-sub {
    font: 400 .68rem 'IBM Plex Mono', monospace;
    color: var(--muted) !important;
    letter-spacing: .08em;
    text-transform: uppercase;
    margin-top: .2rem;
}

.sidebar-label {
    font: 600 .68rem 'IBM Plex Mono', monospace;
    letter-spacing: .14em;
    text-transform: uppercase;
    color: var(--accent) !important;
    margin: 1.2rem 0 .45rem;
}

/* ---------- TYPOGRAPHY ---------- */

h1 {
    font-size: 2rem !important;
    font-weight: 600 !important;
    letter-spacing: -.025em;
    color: var(--text) !important;
    margin-bottom: .3rem !important;
}

.eyebrow {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .82rem !important;
    font-weight: 600 !important;
    letter-spacing: .10em !important;
    text-transform: uppercase !important;
    color: #42c6d4 !important;
    margin-bottom: .55rem !important;
    opacity: 1 !important;
}

.subtitle {
    color: var(--muted);
    font-size: .9rem;
    margin-bottom: 1.6rem;
}

.status {
    color: var(--green);
    font-family: 'IBM Plex Mono', monospace;
    font-size: .7rem;
    font-weight: 600;
    letter-spacing: .08em;
    margin-left: .8rem;
}

/* ---------- SECTION HEADERS ---------- */

.section-label {
    font: 600 .7rem 'IBM Plex Mono', monospace;
    letter-spacing: .15em;
    text-transform: uppercase;
    color: var(--muted);
    border-left: 3px solid var(--accent);
    padding-left: .65rem;
    margin: 1.5rem 0 .75rem;
}

/* ---------- METRIC CARDS ---------- */

[data-testid="stMetric"] {
    background: linear-gradient(
        145deg,
        rgba(18,26,34,.98),
        rgba(14,21,28,.98)
    );

    border: 1px solid var(--line);
    border-radius: 10px;

    padding: 1rem 1.1rem;

    min-height: 112px;

    box-shadow:
        0 8px 24px rgba(0,0,0,.18);

    transition:
        border-color .2s ease,
        transform .2s ease;
}

[data-testid="stMetric"]:hover {
    border-color: #3a4c59;
    transform: translateY(-2px);
}

[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: .72rem !important;
    font-family: 'IBM Plex Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: .08em;
}

[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-size: 1.65rem !important;
    font-weight: 600 !important;
}

[data-testid="stMetricDelta"] {
    font-size: .72rem !important;
}

/* ---------- GENERAL CONTAINERS ---------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 10px;
}

/* ---------- TABS ---------- */

button[data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: .72rem !important;
    letter-spacing: .04em;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: var(--accent) !important;
}

/* ---------- TABLE ---------- */

[data-testid="stDataFrame"] {
    border: 1px solid var(--line);
    border-radius: 9px;
    overflow: hidden;
}

/* ---------- SLIDERS ---------- */

[data-testid="stSlider"] [role="slider"] {
    background: var(--accent);
}

/* ---------- CAPTIONS ---------- */

.stCaption {
    color: var(--muted) !important;
}

/* ---------- SCROLLBAR ---------- */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: var(--bg);
}

::-webkit-scrollbar-thumb {
    background: #2b3a45;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #3b4d5a;
}
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

@st.cache_resource
def train_css_model(cycles):
    css_features = [
        "steam_volume_tonnes",
        "injection_pressure_barg",
        "soak_days"
    ]

    X = cycles[css_features]
    y = cycles["peak_wellbore_temp_C"]

    css_model = RandomForestRegressor(
        n_estimators=160,
        random_state=42
    ).fit(X, y)

    return css_model, css_features
@st.cache_resource
def train_css_production_model(cycles):
    css_features = [
        "steam_volume_tonnes",
        "injection_pressure_barg",
        "soak_days"
    ]

    X = cycles[css_features]
    y = cycles["cycle_oil_produced_bbl"]

    model = RandomForestRegressor(
        n_estimators=160,
        random_state=42
    ).fit(X, y)

    return model
@st.cache_resource
def train_css_sor_model(cycles):
    css_features = [
        "steam_volume_tonnes",
        "injection_pressure_barg",
        "soak_days"
    ]

    X = cycles[css_features]
    y = cycles["steam_oil_ratio_SOR"]

    model = RandomForestRegressor(
        n_estimators=160,
        random_state=42
    ).fit(X, y)

    return model

@st.cache_resource
def train_viscosity_model(daily):
    X = daily[["reservoir_zone_temp_C"]]
    y = daily["oil_viscosity_cP"]

    model = RandomForestRegressor(
        n_estimators=160,
        random_state=42
    ).fit(X, y)

    return model

def optimize_css(cycles, css_model, css_production_model, css_sor_model):
    steam_values = cycles["steam_volume_tonnes"].quantile(
        [0.25, 0.50, 0.75]
    ).tolist()

    pressure_values = cycles["injection_pressure_barg"].quantile(
        [0.25, 0.50, 0.75]
    ).tolist()

    soak_values = sorted(cycles["soak_days"].unique())

    scenarios = []

    for steam in steam_values:
        for pressure in pressure_values:
            for soak in soak_values:

                scenario = pd.DataFrame(
                    [[steam, pressure, soak]],
                    columns=[
                        "steam_volume_tonnes",
                        "injection_pressure_barg",
                        "soak_days"
                    ]
                )

                predicted_temp = float(
                    css_model.predict(scenario)[0]
                )

                predicted_oil = float(
                    css_production_model.predict(scenario)[0]
                )

                predicted_sor = float(
                    css_sor_model.predict(scenario)[0]
                )

                scenarios.append({
                    "steam": steam,
                    "pressure": pressure,
                    "soak": soak,
                    "temperature": predicted_temp,
                    "oil": predicted_oil,
                    "sor": predicted_sor
                })

    result = pd.DataFrame(scenarios)

    # Higher oil production and lower SOR are preferred.
    result["score"] = (
        result["oil"] / result["sor"].clip(lower=0.01)
    )

    return result.sort_values(
        ["score", "oil"],
        ascending=[False, False]
    ).reset_index(drop=True)
@st.cache_data
def build_css_srp_dataset(cycles, daily):
    css = cycles.copy()
    srp = daily.copy()

    css["production_start_date"] = pd.to_datetime(
        css["production_start_date"]
    )

    css["production_end_date"] = (
        css["production_start_date"]
        + pd.to_timedelta(
            css["production_days_actual"] - 1,
            unit="D"
        )
    )

    srp["date"] = pd.to_datetime(srp["date"])

    merged = css.merge(
        srp,
        on="well_id",
        suffixes=("_css", "_srp")
    )

    matched = merged[
        (merged["date"] >= merged["production_start_date"]) &
        (merged["date"] <= merged["production_end_date"])
    ].copy()

    matched["cycle_number"] = matched["cycle_number_css"]

    return matched

@st.cache_resource
def train_css_srp_models(css_srp_data):

    features = [
        "reservoir_zone_temp_C",
        "oil_viscosity_cP",
        "stroke_length_in",
        "spm"
    ]

    X = css_srp_data[features]

    production_model = RandomForestRegressor(
        n_estimators=120,
        random_state=42
    ).fit(
        X,
        css_srp_data["oil_rate_bbl_per_day"]
    )

    risk_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42
    ).fit(
        X,
        css_srp_data["rod_floating_risk_score"]
    )

    return production_model, risk_model

def risk_for(model, scenario):
    risk = float(model.predict(scenario)[0])
    return float(np.clip(risk, 0, 1))


def recommendation(production, health, features, temp, viscosity, risk_limit):
    options = []

    for stroke in [80, 90, 100]:
        for spm in np.arange(2.0, 6.6, 0.2):
            scenario = pd.DataFrame(
                [[temp, viscosity, stroke, spm]],
                columns=features
            )

            oil_prediction = float(production.predict(scenario)[0])
            rod_risk = risk_for(health, scenario)

            options.append({
                "stroke": stroke,
                "spm": round(float(spm), 1),
                "oil": oil_prediction,
                "risk": rod_risk
            })

    candidates = pd.DataFrame(options)

    # Mark operating points that satisfy the accepted rod-risk limit
    candidates["safe"] = candidates["risk"] <= risk_limit

    safe = candidates[candidates["safe"]]

    if not safe.empty:
        # Normal optimization:
        # among safe settings, maximize predicted oil production
        chosen = safe.sort_values(
            "oil",
            ascending=False
        ).iloc[0]

    else:
        # No setting satisfies the risk constraint.
        # Choose the lowest-risk operating point instead of
        # blindly selecting the highest-production point.
        chosen = candidates.sort_values(
            ["risk", "oil"],
            ascending=[True, False]
        ).iloc[0]

    return chosen, candidates


def chart_layout(height=340):
    axis = dict(gridcolor="#1e2a33", zeroline=False, linecolor="#2a3844", tickfont=dict(color="#8b9aa5", size=11), title=dict(font=dict(color="#8b9aa5", size=12)))
    return dict(
        height=height,
        margin=dict(l=12, r=12, t=52, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="IBM Plex Sans", color="#8b9aa5"),
        legend=dict(orientation="h", y=1.1, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color="#c5d0d8", size=12)),
        xaxis=axis,
        yaxis=axis,
    )


def chart_title(text):
    return dict(text=text, font=dict(size=14, color="#e6edf2", family="IBM Plex Sans"))


daily, cycles, failures, wells = load_data()
production, health, features = train_models(daily)
css_model, css_features = train_css_model(cycles)
css_production_model = train_css_production_model(cycles)
css_sor_model = train_css_sor_model(cycles)
viscosity_model = train_viscosity_model(daily)
css_options = optimize_css(
    cycles,
    css_model,
    css_production_model,
    css_sor_model
)

best_css = css_options.iloc[0]
css_srp_data = build_css_srp_dataset(cycles, daily)
css_srp_production, css_srp_health = train_css_srp_models(css_srp_data)

def joint_optimize_css_srp(
    css_steam,
    css_pressure,
    css_soak,
    css_model,
    css_sor_model,
    css_srp_production,
    css_srp_health,
    viscosity_model,
    features,
    risk_limit
):

    # ---------------- CSS PREDICTION ----------------

    css_input = pd.DataFrame(
        [[css_steam, css_pressure, css_soak]],
        columns=[
            "steam_volume_tonnes",
            "injection_pressure_barg",
            "soak_days"
        ]
    )

    predicted_temperature = float(
        css_model.predict(css_input)[0]
    )

    predicted_sor = float(
        css_sor_model.predict(css_input)[0]
    )

    viscosity_input = pd.DataFrame(
       [[predicted_temperature]],
       columns=["reservoir_zone_temp_C"]
    )

    predicted_viscosity = float(
       viscosity_model.predict(viscosity_input)[0]
    )
    # ---------------- SRP OPTIMIZATION ----------------

    srp_scenarios = []

    for stroke in [80, 90, 100]:

        for spm in np.arange(2.0, 6.6, 0.2):

            srp_scenarios.append([
                predicted_temperature,
                predicted_viscosity,
                stroke,
                round(float(spm), 1)
            ])

    srp_scenarios = pd.DataFrame(
        srp_scenarios,
        columns=features
    )

    # Predict oil production
    oil_predictions = css_srp_production.predict(
        srp_scenarios
    )

    # Predict rod-floating risk
    risk_predictions = css_srp_health.predict(
       srp_scenarios
    )

    risk_predictions = np.clip(
       risk_predictions,
      0,
      1
    )

    # Build candidate table
    candidates = srp_scenarios.copy()

    candidates["oil"] = oil_predictions
    candidates["risk"] = risk_predictions

    # Find safe SRP operating points
    safe = candidates[
        candidates["risk"] <= risk_limit
    ]

    if not safe.empty:

        # Among safe settings,
        # choose highest predicted oil production
        chosen = safe.sort_values(
            "oil",
            ascending=False
        ).iloc[0]

    else:

        # If no setting is safe,
        # choose lowest-risk setting
        chosen = candidates.sort_values(
            ["risk", "oil"],
            ascending=[True, False]
        ).iloc[0]

    # ---------------- JOINT SCORE ----------------

    joint_score = float(
        chosen["oil"] /
        (1 + predicted_sor)
    )

    return {
        "steam": float(css_steam),
        "pressure": float(css_pressure),
        "soak": float(css_soak),
        "temperature": predicted_temperature,
        "sor": predicted_sor,
        "viscosity": predicted_viscosity,
        "stroke": float(chosen["stroke_length_in"]),
        "spm": float(chosen["spm"]),
        "oil": float(chosen["oil"]),
        "risk": float(chosen["risk"]),
        "joint_score": joint_score
    }

well_ids = sorted(wells["well_id"].unique())

with st.sidebar:
    st.markdown(
        '<p class="brand">WELLSIGHT</p>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="brand-sub">Operations intelligence · v2.0</p>',
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown(
        '<p class="sidebar-label">Well selection</p>',
        unsafe_allow_html=True
    )

    selected_well = st.selectbox(
        "Asset focus",
        well_ids,
        index=0
    )

    well_daily = daily[
        daily["well_id"] == selected_well
    ].sort_values("date")

    latest = well_daily.iloc[-1]

    # ---------------- CSS CONTROLS ----------------

    st.markdown(
        '<p class="sidebar-label">CSS scenario controls</p>',
        unsafe_allow_html=True
    )

    latest_css = cycles[
        cycles["well_id"] == selected_well
    ].sort_values("cycle_number").iloc[-1]

    css_steam = st.slider(
    "Steam volume (tonnes)",
    float(cycles["steam_volume_tonnes"].min()),
    float(cycles["steam_volume_tonnes"].max()),
    float(latest_css["steam_volume_tonnes"]),
    10.0
)

    css_pressure = st.slider(
        "Injection pressure (barg)",
        5.0,
        40.0,
        float(latest_css["injection_pressure_barg"]),
        0.5
    )

    css_soak = st.slider(
        "Soak period (days)",
        1.0,
        10.0,
        float(latest_css["soak_days"]),
        1.0
    )

    # ---------------- SRP CONTROLS ----------------

    st.markdown(
        '<p class="sidebar-label">SRP scenario controls</p>',
        unsafe_allow_html=True
    )

    temperature = st.slider(
        "Reservoir-zone temperature (°C)",
        40.0,
        200.0,
        float(latest["reservoir_zone_temp_C"]),
        0.5
    )

    viscosity = st.slider(
        "Oil viscosity (cP)",
        50.0,
        4000.0,
        float(latest["oil_viscosity_cP"]),
        10.0
    )
    stroke = st.slider(
      "Stroke length (in)",
      60.0,
      110.0,
      float(latest["stroke_length_in"]),
      5.0
    )

    spm = st.slider(
     "Pump speed (SPM)",
      2.0,
      6.5,
      float(latest["spm"]),
      0.1
    )

    risk_limit_pct = st.slider(
        "Maximum accepted rod risk",
        5,
        60,
        25,
        1,
        format="%d%%"
    )

    risk_limit = risk_limit_pct / 100

    lookback = st.slider(
        "Trend window (days)",
        15,
        min(120, len(well_daily)),
        min(60, len(well_daily))
    )

    st.divider()

    st.caption(
        "CSS controls drive thermal prediction and SOR. "
        "SRP controls drive pump optimization and rod-risk prediction."
    )
joint_result = joint_optimize_css_srp(
    css_steam,
    css_pressure,
    css_soak,
    css_model,
    css_sor_model,
    css_srp_production,
    css_srp_health,
    viscosity_model,
    features,
    risk_limit
)
chosen, options = recommendation(css_srp_production, css_srp_health, features, joint_result["temperature"],
    joint_result["viscosity"], risk_limit)
recommended_scenario = pd.DataFrame(
    [[
        joint_result["temperature"],
        joint_result["viscosity"],
        joint_result["stroke"],
        joint_result["spm"]
    ]],
    columns=features
)

recommended_risk = float(chosen["risk"])

current_scenario = pd.DataFrame(
    [[
        temperature,
        viscosity,
        stroke,
        spm
    ]],
    columns=features
)

current_risk = risk_for(
    css_srp_health,
    current_scenario
)

recommendation_safe = recommended_risk <= risk_limit
joint_risk = float(joint_result["risk"])
predicted_fill = max(0, min(100, 98 - viscosity / 65 - chosen["spm"] * 1.2))
predicted_load = 5 + viscosity / 300 + chosen["spm"] * .5
asset_cycles = cycles[cycles["well_id"] == selected_well]
asset_failures = failures[failures["well_id"] == selected_well]
asset_info = wells[wells["well_id"] == selected_well].iloc[0]
latest_css = asset_cycles.sort_values("cycle_number").iloc[-1]
css_scenario = pd.DataFrame(
    [[
        css_steam,
        css_pressure,
        css_soak
    ]],
    columns=css_features
)

predicted_css_temp = float(
    css_model.predict(css_scenario)[0]
)

predicted_css_sor = float(
    css_sor_model.predict(css_scenario)[0]
)
selected_cycle_srp = css_srp_data[
    (css_srp_data["well_id"] == selected_well) &
    (css_srp_data["cycle_number"] == latest_css["cycle_number"])
]
if not selected_cycle_srp.empty:
    css_srp_temp = selected_cycle_srp["reservoir_zone_temp_C"].mean()
    css_srp_viscosity = selected_cycle_srp["oil_viscosity_cP"].mean()
else:
    css_srp_temp = latest["reservoir_zone_temp_C"]
    css_srp_viscosity = latest["oil_viscosity_cP"]

css_aware_chosen, css_aware_options = recommendation(
    production,
    health,
    features,
    css_srp_temp,
    css_srp_viscosity,
    risk_limit
)

st.html("""
<p style="
    color: #42c6d4 !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    margin: 0 0 12px 0 !important;
    opacity: 1 !important;
">
    FIELD COMMAND &nbsp; / &nbsp; DIGITAL TWIN
</p>
""")
st.title("Well Operations Intelligence")
st.markdown(f'<div class="subtitle">Operating picture for <b>{selected_well}</b> · last telemetry {latest["date"].strftime("%d %b %Y")} <span class="status">● MODEL ONLINE</span></div>', unsafe_allow_html=True)

st.markdown('<div class="section-label">Live operating picture</div>', unsafe_allow_html=True)
m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Oil rate", f"{latest['oil_rate_bbl_per_day']:.1f} bbl/d", f"{latest['water_cut_frac'] * 100:.1f}% water cut", delta_color="off")
production_gain = chosen['oil'] - latest['oil_rate_bbl_per_day']
production_gain_pct = (
    production_gain / latest['oil_rate_bbl_per_day'] * 100
    if latest['oil_rate_bbl_per_day'] != 0
    else 0
)

m2.metric(
    "Scenario output",
    f"{chosen['oil']:.1f} bbl/d",
    f"+{production_gain:.1f} bbl/d ({production_gain_pct:.1f}%)"
)
m3.metric(
    "Current rod risk",
    f"{current_risk * 100:.1f}%",
    f"limit {risk_limit * 100:.0f}%",
    delta_color="inverse"
)
m4.metric("Fillage", f"{predicted_fill:.1f}%", f"{latest['pump_fillage_frac'] * 100:.1f}% observed")
m5.metric("Motor load", f"{predicted_load:.1f} kW", f"{latest['motor_load_kW']:.1f} kW observed")
m6.metric("Recommended", f"{chosen['stroke']:.0f} in / {chosen['spm']:.1f} SPM", "optimized setpoint")

st.markdown(
    '<div class="section-label">CSS thermal state</div>',
    unsafe_allow_html=True
)

css1, css2, css3, css4 = st.columns(4)

css1.metric(
    "Latest CSS cycle",
    f"Cycle {int(latest_css['cycle_number'])}"
)

css2.metric(
    "Steam scenario",
    f"{css_steam:.1f} t"
)

css3.metric(
    "Predicted temperature",
    f"{predicted_css_temp:.1f} °C"
)

css4.metric(
    "Predicted SOR",
    f"{predicted_css_sor:.2f}"
)
if joint_result is not None:

    st.markdown("## Joint CSS + SRP Optimal Scenario")
    st.caption("Recommended operating point based on the joint optimization model")

    css_col, srp_col = st.columns(2, gap="large")

    with css_col:
        with st.container(border=True):
            st.markdown("#### CSS OPTIMIZATION")
            st.caption("Cyclic Steam Stimulation")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Steam",
                    f'{joint_result["steam"]:.1f} t'
                )

                st.metric(
                    "Injection Pressure",
                    f'{joint_result["pressure"]:.1f} barg'
                )

                st.metric(
                    "Soak",
                    f'{joint_result["soak"]:.1f} days'
                )

            with c2:
                st.metric(
                    "Predicted Temperature",
                    f'{joint_result["temperature"]:.1f} °C'
                )

                st.metric(
                    "Predicted SOR",
                    f'{joint_result["sor"]:.2f}'
                )

    with srp_col:
        with st.container(border=True):
            st.markdown("#### SRP OPTIMIZATION")
            st.caption("Sucker Rod Pump")

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Stroke",
                    f'{joint_result["stroke"]:.0f} in'
                )

                st.metric(
                    "SPM",
                    f'{joint_result["spm"]:.1f}'
                )

            with c2:
                st.metric(
                    "Predicted Oil Production",
                    f'{joint_result["oil"]:.1f} bbl/d'
                )

                st.metric(
                    "Predicted Rod-Floating Risk",
                    f'{recommended_risk * 100:.1f}%'
                )

    st.markdown("")

    score = joint_result["joint_score"]

    if score >= 90:
        status = "Excellent"
    elif score >= 75:
        status = "Good"
    else:
        status = "Needs Improvement"

    with st.container(border=True):
        st.markdown("### JOINT OPTIMIZATION SCORE")

        score_col, status_col = st.columns([2, 1])

        with score_col:
            st.metric(
                "Overall Score",
                f"{score:.2f}"
            )

        with status_col:
            st.metric(
                "Assessment",
                status
            )

else:
    st.warning("No feasible joint CSS + SRP scenario found.")

st.markdown(
    '<div class="section-label">Operational advisory</div>',
    unsafe_allow_html=True
)

if current_risk > risk_limit:

    st.warning(
        f"Current operating condition exceeds the {risk_limit * 100:.0f}% rod-risk limit."
    )

    adv1, adv2, adv3 = st.columns(3)

    adv1.metric(
        "Current rod risk",
        f"{current_risk * 100:.1f}%"
    )

    adv2.metric(
        "Recommended rod risk",
        f"{recommended_risk * 100:.1f}%"
    )

    adv3.metric(
        "Expected oil production",
        f"{chosen['oil']:.1f} bbl/d"
    )

    st.info(
        f"Recommended setting: "
        f"{chosen['stroke']:.0f} in / {chosen['spm']:.1f} SPM"
    )

else:

    st.success(
        f"Current operating condition is within the "
        f"{risk_limit * 100:.0f}% rod-risk limit."
    )

    adv1, adv2, adv3 = st.columns(3)

    adv1.metric(
        "Current rod risk",
        f"{current_risk * 100:.1f}%"
    )

    adv2.metric(
        "Recommended rod risk",
        f"{recommended_risk * 100:.1f}%"
    )

    adv3.metric(
        "Expected oil production",
        f"{chosen['oil']:.1f} bbl/d"
    )

    st.info(
        f"Recommended setting: "
        f"{chosen['stroke']:.0f} in / {chosen['spm']:.1f} SPM"
    )

left, center, right = st.columns([1.1, 1.4, 1.1])
with left:
    st.markdown(
        '<div class="section-label">Recommended action</div>',
        unsafe_allow_html=True
    )

    if recommendation_safe:
        action_text = (
            f'<b>Operate at {chosen["stroke"]:.0f} in and '
            f'{chosen["spm"]:.1f} SPM.</b><br>'
            f'Projected {chosen["oil"]:.1f} bbl/d with rod risk at '
            f'{recommended_risk * 100:.1f}%, within the {risk_limit * 100:.0f}% limit.'
        )
    else:
        action_text = (
            f'<b>No safe operating point found under the '
            f'{risk_limit * 100:.0f}% rod-risk limit.</b><br>'
            f'Best available scenario: {chosen["stroke"]:.0f} in and '
            f'{chosen["spm"]:.1f} SPM, projecting {chosen["oil"]:.1f} bbl/d '
            f'with {recommended_risk * 100:.1f}% rod risk.'
        )

    st.markdown(
        f'<div class="callout">{action_text}</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f'<p class="delta-note">'
        f'Scenario delta: '
        f'{temperature - latest["reservoir_zone_temp_C"]:+.1f} °C and '
        f'{viscosity - latest["oil_viscosity_cP"]:+.0f} cP from latest telemetry.'
        f'</p>',
        unsafe_allow_html=True
    )  
with center:
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=current_risk * 100,
        number={"suffix": "%", "font": {"size": 28, "color": "#e6edf2"}},
        title={"text": "CURRENT ROD FLOATING RISK", "font": {"size": 12, "color": "#8b9aa5"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#4a5a66", "tickfont": {"color": "#8b9aa5"}},
            "bar": {"color": "#e09a45", "thickness": 0.72},
            "bgcolor": "#12181f",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 25], "color": "#163028"},
                {"range": [25, 50], "color": "#2a2618"},
                {"range": [50, 100], "color": "#2a1818"},
            ],
            "threshold": {"line": {"color": "#e6edf2", "width": 2}, "value": risk_limit * 100},
        },
    ))
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
    figure.add_trace(go.Scatter(x=trend["date"], y=trend["oil_rate_bbl_per_day"], name="Oil rate", line={"color": "#3db8c5", "width": 2.5}, fill="tozeroy", fillcolor="rgba(61,184,197,.12)"), secondary_y=False)
    figure.add_trace(go.Scatter(x=trend["date"], y=trend["reservoir_zone_temp_C"], name="Temperature", line={"color": "#e09a45", "width": 2}), secondary_y=True)
    figure.add_trace(go.Scatter(x=trend["date"], y=trend["oil_viscosity_cP"], name="Viscosity", line={"color": "#7a8fa3", "width": 1.5, "dash": "dot"}), secondary_y=True)
    figure.update_yaxes(title_text="Oil rate (bbl/d)", secondary_y=False, gridcolor="#1e2a33", linecolor="#2a3844", tickfont=dict(color="#8b9aa5"), title=dict(font=dict(color="#8b9aa5")))
    figure.update_yaxes(title_text="Thermal state / viscosity", secondary_y=True, gridcolor="#1e2a33", linecolor="#2a3844", tickfont=dict(color="#8b9aa5"), title=dict(font=dict(color="#8b9aa5")))
    figure.update_layout(**chart_layout(390), title=chart_title("Recorded decline and fluid behavior"))
    st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    a, b, c = st.columns(3)
    a.metric("Avg. production", f"{trend['oil_rate_bbl_per_day'].mean():.1f} bbl/d")
    b.metric("Avg. efficiency", f"{trend['pump_volumetric_efficiency'].mean() * 100:.1f}%")
    c.metric("Avg. polished-rod load", f"{trend['polished_rod_load_lbs'].mean():,.0f} lb")
with tab2:
    envelope = options.copy()
    envelope["risk_pct"] = envelope["risk"] * 100
    heat = go.Figure(go.Scatter(x=envelope["spm"], y=envelope["oil"], mode="markers", marker={"size": 9, "color": envelope["risk_pct"], "colorscale": [[0, "#3db8c5"], [.25, "#c9a227"], [1, "#c45c4a"]], "colorbar": {"title": {"text": "Risk %", "font": {"color": "#8b9aa5"}}, "tickfont": {"color": "#8b9aa5"}}, "line": {"width": .5, "color": "#0c1116"}}, text=[f"{s:.0f} in / {v:.1f} SPM" for s, v in zip(envelope["stroke"], envelope["spm"])], hovertemplate="%{text}<br>Oil: %{y:.1f} bbl/d<extra></extra>"))
    heat.add_trace(go.Scatter(x=[chosen["spm"]], y=[chosen["oil"]], mode="markers", marker={"size": 16, "color": "#f0d78c", "symbol": "star", "line": {"width": 1, "color": "#0c1116"}}, name="Recommended"))
    heat.update_layout(**chart_layout(390), title=chart_title("AI-tested pump operating envelope"), xaxis_title="Speed (SPM)", yaxis_title="Predicted oil rate (bbl/d)")
    st.plotly_chart(heat, use_container_width=True, config={"displayModeBar": False})
    st.caption(f"{len(envelope[envelope['safe']])} of {len(envelope)} tested settings satisfy the {risk_limit * 100:.0f}% risk limit.")
with tab3:
    c1, c2 = st.columns([1.3, 1])
    with c1:
        cycle = asset_cycles.sort_values("cycle_number")
        cycle_fig = go.Figure()
        cycle_fig.add_trace(go.Bar(x=cycle["cycle_number"], y=cycle["cycle_oil_produced_bbl"], name="Cycle oil", marker_color="#3d9a8f", marker_line_width=0))
        cycle_fig.add_trace(go.Scatter(x=cycle["cycle_number"], y=cycle["steam_oil_ratio_SOR"] * cycle["cycle_oil_produced_bbl"].max() / 2, name="SOR (scaled)", line={"color": "#e09a45", "width": 2.5}, mode="lines+markers", marker={"size": 6, "color": "#e09a45"}))
        cycle_fig.update_layout(**chart_layout(360), title=chart_title("CSS cycle yield and steam efficiency"), xaxis_title="Cycle", yaxis_title="Oil produced (bbl)")
        st.plotly_chart(cycle_fig, use_container_width=True, config={"displayModeBar": False})
    with c2:
        mode_counts = asset_failures["failure_mode"].value_counts()
        failure_fig = go.Figure(go.Bar(x=mode_counts.values, y=mode_counts.index, orientation="h", marker_color="#c45c4a", marker_line_width=0))
        failure_fig.update_layout(**chart_layout(360), title=chart_title(f"Failure history · {len(asset_failures)} events"), xaxis_title="Events")
        st.plotly_chart(failure_fig, use_container_width=True, config={"displayModeBar": False})

st.markdown('<div class="section-label">Signal register</div>', unsafe_allow_html=True)
signal_columns = ["date", "day_in_cycle", "oil_rate_bbl_per_day", "water_cut_frac", "fluid_level_pct", "pump_volumetric_efficiency", "polished_rod_load_lbs", "rod_floating_risk_score"]
signals = well_daily.tail(12)[signal_columns].copy()
signals.columns = ["Date", "Cycle day", "Oil bbl/d", "Water cut", "Fluid level %", "Vol. efficiency", "Rod load lb", "Risk score"]
signals["Date"] = signals["Date"].dt.strftime("%d %b %Y")
st.dataframe(signals.sort_values("Date", ascending=False), hide_index=True, use_container_width=True, column_config={"Water cut": st.column_config.NumberColumn(format="%.1f%%"), "Vol. efficiency": st.column_config.NumberColumn(format="%.1f%%"), "Risk score": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
