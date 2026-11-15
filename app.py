import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import urllib.request, json
import geopandas as gpd
from shapely.geometry import Point

st.set_page_config(
    page_title="FireSignal",
    page_icon="None",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; background-color: #f0f4f8; color: #1a2332; }
    .main { background-color: #f0f4f8; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
    .header-wrap { background: linear-gradient(135deg, #1e3a5f 0%, #2563a8 60%, #3b82c4 100%); border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(37,99,168,0.18); }
    .header-title { font-family: 'Sora', sans-serif; font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; line-height: 1.15; margin-bottom: 0.4rem; }
    .header-sub { font-size: 0.9rem; color: #a8c8e8; font-weight: 300; letter-spacing: 0.8px; text-transform: uppercase; }
    .metric-card { background: #ffffff; border: 1px solid #dbe8f5; border-radius: 12px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 2px 8px rgba(37,99,168,0.07); }
    .metric-value { font-family: 'Sora', sans-serif; font-size: 1.8rem; font-weight: 700; color: #2563a8; display: block; }
    .metric-label { font-size: 0.7rem; color: #6b8cae; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
    .section-title { font-family: 'Sora', sans-serif; font-size: 1.2rem; font-weight: 600; color: #1e3a5f; border-bottom: 2px solid #dbe8f5; padding-bottom: 0.5rem; margin-bottom: 1.2rem; margin-top: 2rem; }
    .about-box { background: #ffffff; border: 1px solid #dbe8f5; border-left: 4px solid #2563a8; border-radius: 10px; padding: 1.5rem; font-size: 0.88rem; color: #4a6080; line-height: 1.9; box-shadow: 0 2px 8px rgba(37,99,168,0.05); }
    .tag { display: inline-block; background: #e8f0fb; color: #2563a8; font-size: 0.72rem; font-weight: 500; padding: 0.2rem 0.6rem; border-radius: 20px; margin-right: 0.3rem; }
</style>
""", unsafe_allow_html=True)

DRIVE = "." 

@st.cache_resource
def load_models():
    with open(f"{DRIVE}/baseline_forecast_model.pkl", "rb") as f:
        base = pickle.load(f)
    with open(f"{DRIVE}/spread_model.pkl", "rb") as f:
        spread = pickle.load(f)
    return base, spread

@st.cache_data
def load_data():
    sites = pd.read_csv(f"{DRIVE}/sites_with_population.csv")
    kbdi = pd.read_csv(f"{DRIVE}/kbdi_features.csv")
    fires = pd.read_csv(f"{DRIVE}/fires_signal.csv")
    return sites, kbdi, fires

base_bundle, spread_bundle = load_models()
sites, kbdi, fires_meta = load_data()

st.markdown("""
<div class="header-wrap">
    <div class="header-title">FireSignal</div>
    <div class="header-sub">Australian Fire Ignition Forecast &nbsp;·&nbsp; Spatiotemporal Model &nbsp;·&nbsp; Based on harmonised NASA MODIS/VIIRS data</div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1rem 1.2rem;margin-bottom:1.6rem;
box-shadow:0 2px 8px rgba(37,99,168,0.05);font-size:0.92rem;color:#4a6080;line-height:1.8;'>
FireSignal estimates the probability of subsequent fire activity across Australia, incorporating spatial and temporal relationships 
in harmonised satellite observations. It is designed to provide an additional information layer to support fire agencies in resource allocation and planning.
</div>
""", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
metrics = [
    ("0.854", "Model AUC"),
    ("+0.044", "Contagion Impact"),
    ("249,246", "Grid Cells Analysed"),
    ("486,255", "NASA Detections Harmonised"),
]
for col, (val, label) in zip([c1, c2, c3, c4], metrics):
    col.markdown(f"""
    <div class="metric-card">
        <span class="metric-value">{val}</span>
        <span class="metric-label">{label}</span>
    </div>""", unsafe_allow_html=True)

st.markdown("<div style='margin-top: 2rem'></div>", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>What This App Does</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
This tool estimates the probability of fire ignition using weather, geography, drought indicators, and historical fire dynamics. 
It acts as a triage map, identifying where attention may be needed most urgently.
</div>

<div style='font-size:0.82rem; color:#1e3a5f; font-family:Sora,sans-serif; font-weight:600; margin-bottom:0.4rem'>Why This Matters</div>
<div style='font-size:0.78rem; color:#4a6080; line-height:1.8; margin-bottom:1.2rem'>
Standard indices indicate when conditions are dangerous. FireSignal demonstrates that harmonised satellite observations contain 
additional predictive information about where fire activity will occur next, driven by spatial and temporal dependence.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Model Validation", "Forecast Map", "Fire Regimes"])

with tab1:
    st.markdown('<div class="section-title">Rigorous Model Evaluation</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background:#e8f0fb;border-radius:10px;padding:1rem 1.4rem;margin-bottom:1.2rem;font-size:0.86rem;color:#1e3a5f;line-height:1.8;'>
    We aggregated NASA FIRMS detections into a 1-degree grid, generating 249,246 daily records. 
    We applied a strict chronological train/test split (Train: Jan-Oct, Test: Nov-Dec).
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**1. Ablation Test**")
    st.markdown("""
    To isolate the predictive value of spatial dependence, we compared two models:
    *   **Base Model (Geography + Weather):** AUC 0.810
    *   **FireSignal (+ Neighbour-Fire Feature):** AUC 0.854
    *   *Result:* A paired bootstrap confirmed that adding the neighbour-fire feature improved AUC by +0.044 (95% CI: +0.041 to +0.047, p < 0.001).
    """)

    st.markdown("**2. Permutation Importance**")
    st.markdown("""
    Rather than relying on default feature importance, we shuffled individual features to measure their impact on model performance:
    *   `neighbor_fire_yesterday`: **0.144**
    *   `longitude`: 0.067
    *   `latitude`: 0.067
    """)

    st.markdown("""
    <div style='background:#ffffff;border:1px solid #dbe8f5;border-left:4px solid #2563a8;border-radius:10px;
    padding:1.2rem 1.4rem;margin-top:1.5rem;box-shadow:0 2px 8px rgba(37,99,168,0.05);font-size:0.85rem;
    color:#4a6080;line-height:1.8;'>
    <b>Conclusion:</b> Fire occurrence is not solely a function of weather or location. It is measurably influenced by recent nearby fire activity.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Historical Forecast Demonstration (November 15, 2020)</div>', unsafe_allow_html=True)
    st.markdown("Predicted probability of ignition across the Australian grid using historical data.")
    
    # 1. Generate the grid
    lat_grid = np.arange(-44, -9, 1)
    lon_grid = np.arange(113, 155, 1)
    grid_cells = [{'lat_cell': float(lat), 'lon_cell': float(lon)} for lat in lat_grid for lon in lon_grid]
    forecast_df = pd.DataFrame(grid_cells)
    forecast_df['doy'] = pd.to_datetime('2020-11-15').dayofyear
    
    # 2. Load features
    sites['lat_cell'] = (sites['latitude'] // 1).astype(int)
    sites['lon_cell'] = (sites['longitude'] // 1).astype(int)
    sites_agg = sites.groupby(['lat_cell', 'lon_cell']).agg({
        'log_population_5km': 'mean', 'dist_to_powerline_km': 'mean'}).reset_index()

    kbdi = kbdi.merge(fires_meta[['site_id','latitude','longitude']], on='site_id')
    kbdi['lat_cell'] = (kbdi['latitude'] // 1).astype(int)
    kbdi['lon_cell'] = (kbdi['longitude'] // 1).astype(int)
    kbdi_agg = kbdi.groupby(['lat_cell', 'lon_cell'])['kbdi_max'].mean().reset_index()

    forecast_df = forecast_df.merge(sites_agg, on=['lat_cell', 'lon_cell'], how='left')
    forecast_df = forecast_df.merge(kbdi_agg, on=['lat_cell', 'lon_cell'], how='left')
    
    # 3. Generate Predictions
    FEATURES = base_bundle['features']
    for c in FEATURES:
        if c not in forecast_df.columns:
            forecast_df[c] = 0
        forecast_df[c] = forecast_df[c].fillna(forecast_df[c].median())

    forecast_df['risk'] = base_bundle['model'].predict_proba(forecast_df[FEATURES])[:,1]

    # 4. Render using PyDeck for a professional look
    # Filter out low risk and create a color column
    active_risk = forecast_df[forecast_df['risk'] > 0.15].copy()
    
    # PyDeck requires RGBA arrays for colors
    def get_color(risk):
        if risk > 0.75: return [220, 38, 38, 220] # Deep Red
        if risk > 0.50: return [234, 88, 12, 200]  # Orange
        if risk > 0.30: return [234, 179, 8, 180]  # Yellow
        return [37, 99, 168, 120]                  # Blue (Low risk)
        
    active_risk['color'] = active_risk['risk'].apply(get_color)
    active_risk['lat_center'] = active_risk['lat_cell'] + 0.5
    active_risk['lon_center'] = active_risk['lon_cell'] + 0.5

    layer = pdk.Layer(
        "ScatterplotLayer",
        active_risk,
        get_position=["lon_center", "lat_center"],
        get_fill_color="color",
        get_radius=45000, # 45km radius to cover the 1-degree grid
        pickable=True,
        opacity=0.8,
        stroked=False,
    )

    view_state = pdk.ViewState(
        latitude=-27.0,
        longitude=134.0,
        zoom=3.5,
        pitch=0,
    )

    tooltip = {
        "html": "<b>Grid Cell</b><br/>Lat: {lat_cell}<br/>Lon: {lon_cell}<br/><b>Risk: {risk}</b>",
        "style": {"backgroundColor": "#1e3a5f", "color": "white"}
    }

    st.pydeck_chart(pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
    ))

with tab3:
    st.markdown('<div class="section-title">Fire Regime Analysis</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div style='background:#ffffff;border:1px solid #dbe8f5;border-left:4px solid #2563a8;border-radius:10px;
    padding:1.2rem 1.4rem;margin-bottom:1.2rem;box-shadow:0 2px 8px rgba(37,99,168,0.05);font-size:0.85rem;
    color:#4a6080;line-height:1.8;'>
    Our research identified differences between human-associated and naturally caused fire activity:
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;
        margin-bottom:1rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <div style='font-family:Sora,sans-serif;font-size:1rem;font-weight:700;color:#1e3a5f;margin-bottom:0.8rem'>
                Human-Associated Regime
            </div>
            <div style='font-size:0.84rem;color:#4a6080;line-height:1.85;'>
                Fires near infrastructure. Ignition probability is higher, but severity remains lower due to rapid suppression response.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;
        margin-bottom:1rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <div style='font-family:Sora,sans-serif;font-size:1rem;font-weight:700;color:#1e3a5f;margin-bottom:0.8rem'>
                Remote / Natural Regime
            </div>
            <div style='font-size:0.84rem;color:#4a6080;line-height:1.85;'>
                Fires in remote areas. These present significant monitoring and response challenges, and are highly correlated with accumulated drought.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#e8f0fb;border-radius:10px;padding:1rem 1.4rem;margin-top:1rem;font-size:0.86rem;color:#1e3a5f;line-height:1.8;'>
    <b>Implication:</b> Standard fire indices treat these regimes identically. FireSignal provides an additional information layer to distinguish where decision-makers may need to allocate resources.
    </div>
    """, unsafe_allow_html=True)

st.markdown('<div class="section-title">About This Model</div>', unsafe_allow_html=True)
st.markdown("""
<div class="about-box">
<b style='color:#1e3a5f'>Methodology</b><br>
This model utilises XGBoost trained on harmonised MODIS/VIIRS satellite data, weather indices, population density, and infrastructure proximity across 249,246 daily records in 2020.
It is designed to estimate the probability of <b>subsequent fire activity</b>, providing a predictive signal for where attention may be needed.<br><br>

<b style='color:#1e3a5f'>What This New Layer Adds</b><br>
By incorporating spatial and temporal dependence, FireSignal demonstrated that a neighbouring-fire feature improved AUC from 0.810 to 0.854, proving that satellite data contains additional predictive information beyond weather and geography.<br><br>

<b style='color:#1e3a5f'>Key Limitation</b><br>
The model predicts <b>ignition probability</b>, not severity. Predicting fire size (FRP) failed (CV R² -0.11) because size is driven by wind direction and suppression timing, which satellites cannot observe.<br><br>

<span class="tag">XGBoost</span>
<span class="tag">Spatiotemporal</span>
<span class="tag">Ablation</span>
<span class="tag">MODIS / VIIRS</span>
<span class="tag">Streamlit</span>
</div>
""", unsafe_allow_html=True)
