import streamlit as st
import pandas as pd
import numpy as np
import pickle
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
    
    /* Header Box */
    .header-wrap { background: #1e3a5f; border-radius: 16px; padding: 2.5rem 3rem; margin-bottom: 2rem; box-shadow: 0 4px 24px rgba(30,58,95,0.3); }
    .header-title { font-family: 'Sora', sans-serif; font-size: 2.4rem; font-weight: 700; color: #ffffff; letter-spacing: -0.5px; line-height: 1.15; margin-bottom: 0.4rem; }
    .header-sub { font-size: 0.9rem; color: #a8c8e8; font-weight: 500; letter-spacing: 0.8px; text-transform: uppercase; }
    
    /* Metric Cards */
    .metric-card { background: #ffffff; border: 1px solid #dbe8f5; border-radius: 12px; padding: 1.2rem 1rem; text-align: center; box-shadow: 0 2px 8px rgba(37,99,168,0.07); }
    .metric-value { font-family: 'Sora', sans-serif; font-size: 1.8rem; font-weight: 700; color: #2563a8; display: block; }
    .metric-label { font-size: 0.7rem; color: #6b8cae; text-transform: uppercase; letter-spacing: 1px; margin-top: 0.2rem; }
    
    /* Section Titles */
    .section-title { font-family: 'Sora', sans-serif; font-size: 1.2rem; font-weight: 600; color: #1e3a5f; border-bottom: 2px solid #dbe8f5; padding-bottom: 0.5rem; margin-bottom: 1.2rem; margin-top: 2rem; }
    
    /* Info Boxes */
    .info-box { background: #1e3a5f; border-radius: 10px; padding: 1rem 1.4rem; margin-bottom: 1.2rem; font-size: 0.86rem; color: #ffffff; line-height: 1.8; }
    .info-box b { color: #a8c8e8; }
    
    /* About Box */
    .about-box { background: #1e3a5f; border: 1px solid #dbe8f5; border-left: 4px solid #3b82c4; border-radius: 10px; padding: 1.5rem; font-size: 0.88rem; color: #ffffff; line-height: 1.9; box-shadow: 0 2px 8px rgba(37,99,168,0.05); }
    .about-box b { color: #a8c8e8; }
    
    /* Tags */
    .tag { display: inline-block; background: #2563a8; color: #ffffff; font-size: 0.72rem; font-weight: 500; padding: 0.2rem 0.6rem; border-radius: 20px; margin-right: 0.3rem; }
    
    /* Text Selection Highlight */
    ::selection { background: #3b82c4; color: #ffffff; }
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
    <div class="header-sub">Australian Fire Ignition Forecast &nbsp;|&nbsp; Based on harmonised NASA MODIS/VIIRS data</div>
</div>
""", unsafe_allow_html=True)

# --- THE BLACK SUMMER IMPACT BLOCK ---
st.markdown("""
<div style='background:#ffffff; border-left: 6px solid #dc2626; border-radius: 10px; padding: 1.5rem 1.8rem; margin-bottom: 1.5rem; box-shadow: 0 2px 8px rgba(220,38,38,0.1);'>
    <div style='font-family: Sora, sans-serif; font-size: 1.1rem; font-weight: 700; color: #dc2626; margin-bottom: 0.8rem;'>
        Trained on the data from the 2019-2020 Black Summer.
    </div>
    <div style='font-size: 0.9rem; color: #4a6080; line-height: 1.8;'>
        We built this model using the lessons from Australia's most catastrophic fire season. The human cost was devastating:
        <ul style='margin-top: 0.5rem; margin-bottom: 0.8rem;'>
            <li><b>33 people</b> lost their lives.</li>
            <li><b>Over 17 to 24 million hectares</b> of land burned.</li>
            <li><b>More than 3,000 houses</b> were destroyed.</li>
            <li>An estimated <b>3 billion animals</b> were killed or displaced.</li>
        </ul>
        <b style='color: #1e3a5f;'>We have the data. We have the technology. We need to do better.</b>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
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
Standard indices indicate when conditions are dangerous. FireSignal shows that harmonised satellite observations contain 
additional predictive information about where fire activity will occur next, driven by spatial and temporal dependence.
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["How We Tested It", "Forecast Map", "Where Fires Start", "Data and Limitations"])

with tab1:
    st.markdown('<div class="section-title">How We Tested It</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    We grouped NASA FIRMS detections into a 1 degree grid (roughly 100km squares), creating 249,246 daily records. 
    We trained the model on January to October and tested it on November to December to prevent data leakage.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**In Plain English**")
    st.markdown("""
    Standard fire models rely on weather. They tell you *"it is hot and dry here, so fire is possible."* 
    But they do not tell you **where** the next fire will start.<br><br>
    
    Our research shows that fire acts like a spreading infection. If a fire is burning in one area today, 
    the probability of a new fire starting in the neighbouring area tomorrow skyrockets. By adding a feature 
    that simply asks the model *"Was there a fire next door yesterday?"*, our ability to predict where 
    the next fire will happen improved dramatically.<br><br>
    
    **The Analogy:** Weather tells you if the grass is dry enough to catch a spark. Spatial contagion tells you 
    where the sparks are currently flying. You need both to predict the next fire.
    """, unsafe_allow_html=True)

    st.markdown("**1. Ablation Test**")
    st.markdown("""
    To isolate the predictive value of spatial dependence, we compared two identical models:
    
    | Model | Features | AUC | PR-AUC |
    | :--- | :--- | :--- | :--- |
    | **Base Model** | Geography, Weather, Drought | 0.810 | 0.421 |
    | **FireSignal** | Base + Neighbour Fire | **0.854** | **0.543** |
    
    A paired bootstrap on identical test rows (2,000 iterations) confirmed that adding the neighbour fire feature improved AUC by **+0.044** (95% CI: +0.041 to +0.047, p < 0.001).
    """)

    st.markdown("**2. Permutation Importance**")
    st.markdown("""
    Rather than relying on default feature importance (which can be biased), we shuffled individual features to measure their exact impact:
    *   Neighbour fire yesterday: **0.144**
    *   Longitude: 0.067
    *   Latitude: 0.067
    """)

    st.markdown("**3. Does the model generalise?**")
    st.markdown("""
    We stress tested the model to ensure it was not just memorising the 2020 data:
    
    | Validation Test | Description | Result |
    | :--- | :--- | :--- |
    | Forward in time | Train 2019, Test 2020 | AUC **0.843** |
    | Geographic holdout | Median of 4 spatial splits | AUC **0.790** |
    | New ignitions only | Removed ongoing fires | AUC **0.697** |
    | Severity model | Predict fire size (FRP) | CV R2 **-0.11** |
    """)
    
    st.markdown("""
    <div class="info-box" style="border-left: 4px solid #dc2626;">
    <b>Conclusion and Limitations:</b> Fire occurrence is measurably influenced by recent nearby fire activity. 
    However, we also tested whether the system could predict fire <b>severity</b> (size). It performed poorly (CV R2 -0.11). 
    We report this because understanding the limitations of satellite derived predictions is just as important as understanding their potential.
    </div>
    """, unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="section-title">Forecast Map (November 15, 2020)</div>', unsafe_allow_html=True)
    
    st.markdown("""
    NSW was predicted to be the most warned state, so we decided to focus the app on it showcasing its results. 
    Hover over regions to see the driving factors.<br><br>
    
    <span style="font-size:0.8rem; color:#6b8cae;"><i>Note on map boundaries: Streamlit Community Cloud has strict memory limits. 
    To keep the app fast and responsive, we have used a simplified spatial projection of the LGA boundaries. 
    This causes slight visual distortions in the proportions of the largest LGAs. A production deployment on a dedicated server 
    would render the full-resolution 44MB GeoJSON.</i></span>
    """, unsafe_allow_html=True)

    try:
        with open(f"{DRIVE}/australia_final_colored.geojson", "r") as f:
            lga_geojson = json.load(f)

        # Update populations manually to fix the log conversion
        real_populations = {
            "Tenterfield": "4,066", "Inverell": "16,483", "Moree Plains": "13,159",
            "Walcha": "3,092", "MidCoast": "90,000", "Tamworth Regional": "59,663",
            "Upper Hunter Shire": "14,100", "Cessnock": "55,560", "Narrabri": "13,100",
            "Clarence Valley": "50,000", "Lachlan Valley": "6,000"
        }

        for feature in lga_geojson['features']:
            name = feature['properties']['LGA_NAME_2021']
            
            # Fix population
            if name in real_populations:
                feature['properties']['pop'] = real_populations[name]
            elif feature['properties'].get('status') == "Low Risk":
                feature['properties']['pop'] = "N/A (Safe)"
                
            # Fill empty no-data regions with navy
            if feature['properties'].get('status') == "No Data":
                feature['properties']['color'] = [30, 58, 95, 180]
                feature['properties']['status'] = "Low Risk"

        layer = pdk.Layer(
            "GeoJsonLayer",
            lga_geojson,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.color",
            get_line_color=[255, 255, 255, 80],
            line_width_min_pixels=0.5,
        )
        
        view_state = pdk.ViewState(
            latitude=-30.0,
            longitude=151.5,
            zoom=5.5,
            pitch=0
        )
        
        tooltip = {
            "html": "<div style='font-size:14px; margin-bottom: 5px;'><b>{LGA_NAME_2021}</b></div>"
                    "<div style='font-size:12px;'>Status: <b>{status}</b></div>"
                    "<div style='font-size:16px; margin-bottom: 10px;'>Risk: <b>{risk_val}</b></div>"
                    "<hr style='margin:4px 0; border-color:#4a6080'>"
                    "<div style='font-size:11px;'><b>Model Inputs:</b><br/>"
                    "Drought (KBDI): {kbdi}<br/>"
                    "LGA Population: {pop}<br/>"
                    "Dist to Powerline: {power}</div>",
            "style": {"backgroundColor": "#1e3a5f", "color": "white", "fontSize": "12px", "padding": "10px", "borderRadius": "6px", "width": "250px"}
        }

        st.pydeck_chart(pdk.Deck(
            layers=[layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        ))
        
        # Visual Legend Box
        st.markdown("""
        <div style='background:#ffffff; border-radius:10px; padding:1.2rem; margin-top:1.5rem; box-shadow: 0 2px 8px rgba(37,99,168,0.05); display:flex; justify-content:space-between; align-items:center;'>
            <div style='display:flex; align-items:center; gap: 10px;'>
                <div style='width:20px; height:20px; background:#dc2626; border-radius:4px;'></div>
                <span style='font-size:0.85rem; color:#4a6080;'>Extreme Risk (>75%)</span>
            </div>
            <div style='display:flex; align-items:center; gap: 10px;'>
                <div style='width:20px; height:20px; background:#ea580c; border-radius:4px;'></div>
                <span style='font-size:0.85rem; color:#4a6080;'>High Risk (>60%)</span>
            </div>
            <div style='display:flex; align-items:center; gap: 10px;'>
                <div style='width:20px; height:20px; background:#eab308; border-radius:4px;'></div>
                <span style='font-size:0.85rem; color:#4a6080;'>Moderate Risk (>40%)</span>
            </div>
            <div style='display:flex; align-items:center; gap: 10px;'>
                <div style='width:20px; height:20px; background:#1e3a5f; border-radius:4px;'></div>
                <span style='font-size:0.85rem; color:#4a6080;'>Low Risk</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanatory Note
        st.markdown("""
        <div class="info-box" style="border-left: 4px solid #2563a8; margin-top: 1rem;">
        <b>How to read this map:</b> The model combines weather, geography, and human data to predict risk. 
        For example, <b>Tenterfield</b> is extremely dry (KBDI 195) and remote, meaning a lightning strike would cause a fire 
        that authorities cannot easily reach. This is why it carries an Extreme Risk rating.
        </div>
        """, unsafe_allow_html=True)
        
    except Exception as e:
        st.error(f"Map Error: {e}")

with tab3:
    st.markdown('<div class="section-title">Where Fires Start</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    Our research identified differences between human associated and naturally caused fire activity:
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;
        margin-bottom:1rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <div style='font-family:Sora,sans-serif;font-size:1rem;font-weight:700;color:#1e3a5f;margin-bottom:0.8rem'>
                Human Associated
            </div>
            <div style='font-size:0.84rem;color:#4a6080;line-height:1.85; margin-bottom: 0.8rem;'>
                Fires near infrastructure. Ignition probability is higher, but severity remains lower due to rapid suppression response.
            </div>
            <div style='background:#f0f4f8;border-radius:8px;padding:0.8rem;font-size:0.8rem;color:#1e3a5f;'>
                <b>Avg distance to road:</b> 232 metres<br>
                <b>Avg distance to fire station:</b> 6.5 km<br>
                <b>Example:</b> Accidental ignitions from vehicles, machinery, or arson near general population.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;
        margin-bottom:1rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <div style='font-family:Sora,sans-serif;font-size:1rem;font-weight:700;color:#1e3a5f;margin-bottom:0.8rem'>
                Remote / Natural
            </div>
            <div style='font-size:0.84rem;color:#4a6080;line-height:1.85; margin-bottom: 0.8rem;'>
                Fires in remote areas. These present significant monitoring and response challenges, and are highly correlated with accumulated drought.
            </div>
            <div style='background:#f0f4f8;border-radius:8px;padding:0.8rem;font-size:0.8rem;color:#1e3a5f;'>
                <b>Avg distance to road:</b> 958 metres<br>
                <b>Avg distance to fire station:</b> 14.2 km<br>
                <b>Fire size:</b> 5.7x larger on average than human fires.
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="info-box" style="border-left: 4px solid #dc2626;">
    <b>The Catastrophic Impact:</b> While human activity causes the vast majority of fire ignitions near towns, our analysis of 1,021 fires showed that <b>98% of the largest, most catastrophic fires</b> were lightning associated. Standard fire indices treat these two scenarios as identical. FireSignal provides the data to distinguish them.
    </div>
    """, unsafe_allow_html=True)

with tab4:
    st.markdown('<div class="section-title">Data and Limitations</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Data Sources")
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <ul style='font-size:0.85rem;color:#4a6080;line-height:1.8;margin-bottom:0;'>
                <li><b>NASA FIRMS (MODIS and VIIRS):</b> 486,255 harmonised active fire detections for 2020, and 1.89M for 2019.</li>
                <li><b>Open Meteo:</b> 54,300 daily weather records (temperature, rainfall, wind, ET0).</li>
                <li><b>WorldPop (2020):</b> 1km resolution population density raster.</li>
                <li><b>OpenStreetMap:</b> 19,759 powerline network features.</li>
                <li><b>Queensland Government:</b> 8,784 hourly PM2.5 air quality records.</li>
                <li><b>Mulgar (Monash University):</b> 1,021 cause attributed fires.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Model Limitations")
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <ul style='font-size:0.85rem;color:#4a6080;line-height:1.8;margin-bottom:0;'>
                <li><b>Severity Prediction Failure:</b> We cannot predict fire <i>size</i>. The severity model achieved an R2 of -0.11 because size is dictated by wind direction and suppression timing, which satellites cannot observe.</li>
                <li><b>Resolution Limits:</b> The model operates on a 100km grid. It cannot predict a fire on a specific street corner.</li>
                <li><b>Single Year Base:</b> The primary model is trained on the extreme 2020 Black Summer. It requires calibration for normal fire years.</li>
                <li><b>Active Fire vs Burned Area:</b> We count thermal anomalies, not hectares burned.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("### Future Improvements")
        st.markdown("""
        <div style='background:#ffffff;border:1px solid #dbe8f5;border-radius:12px;padding:1.4rem 1.6rem;box-shadow:0 2px 8px rgba(37,99,168,0.06);'>
            <ul style='font-size:0.85rem;color:#4a6080;line-height:1.8;margin-bottom:0;'>
                <li><b>Live API Integration:</b> The model is currently a historical demonstration. The next step is to connect it to a live daily API that pulls the last 24 hours of FIRMS data and Open Meteo forecasts to generate a real time morning briefing map.</li>
                <li><b>Live Weather Forecasts:</b> Replace the historical day of year proxy with actual 7 day weather forecasts to improve predictive accuracy.</li>
                <li><b>Higher Resolution Grid:</b> Downscale from 100km cells to 25km cells to provide more localised, actionable intelligence for local councils.</li>
                <li><b>Fuel Moisture (SMAP):</b> Integrate NASA SMAP soil moisture data to replace the mathematical proxy for fuel dryness.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
