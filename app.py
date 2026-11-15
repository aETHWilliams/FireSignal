import streamlit as st
import pandas as pd, numpy as np
import pickle, matplotlib.pyplot as plt
import os

st.set_page_config(page_title="FireSignal", layout="wide")
DRIVE = "." 

@st.cache_resource
def load_all():
    with open(f"{DRIVE}/baseline_forecast_model.pkl", "rb") as f:
        base = pickle.load(f)
    with open(f"{DRIVE}/spread_model.pkl", "rb") as f:
        spread = pickle.load(f)
    return base, spread

base_bundle, spread_bundle = load_all()

st.title("FireSignal: Australian Fire Ignition Forecast")
st.markdown("Weather-based indices indicate when conditions are dangerous. FireSignal estimates the probability of subsequent fire activity across Australia, incorporating spatial and temporal relationships in harmonised satellite observations.")

tab1, tab2, tab3 = st.tabs(["Model Validation", "Forecast Map", "Fire Regimes"])

with tab1:
    st.subheader("Rigorous Model Evaluation")
    st.markdown("""
    We harmonised NASA MODIS and VIIRS detections into a 1-degree spatial grid, generating 249,246 daily records. We applied a strict chronological train/test split (Train: Jan-Oct, Test: Nov-Dec).
    
    **1. Ablation Test**
    To isolate the predictive value of spatial dependence, we compared two models:
    *   **Base Model (Geography + Weather):** AUC 0.810
    *   **FireSignal (+ Neighbour-Fire Feature):** AUC 0.854
    
    *Result:* A paired bootstrap confirmed that adding the neighbour-fire feature improved AUC by +0.044 (95% CI: +0.041 to +0.047, p < 0.001).
    
    **2. Permutation Importance**
    Rather than relying on default feature importance, we shuffled individual features to measure their impact on model performance:
    *   `neighbor_fire_yesterday`: **0.144**
    *   `longitude`: 0.067
    *   `latitude`: 0.067
    
    *Conclusion:* Fire occurrence is not solely a function of weather or location. It is measurably influenced by recent nearby fire activity.
    """)

with tab2:
    st.subheader("Historical Forecast Demonstration (November 15, 2020)")
    st.markdown("Predicted probability of ignition across the Australian grid using historical data.")
    
    # 1. Create the grid
    lat_grid = np.arange(-44, -9, 1)
    lon_grid = np.arange(113, 155, 1)
    grid_cells = [{'lat_cell': int(lat), 'lon_cell': int(lon)} for lat in lat_grid for lon in lon_grid]
    forecast_df = pd.DataFrame(grid_cells)
    forecast_df['doy'] = pd.to_datetime('2020-11-15').dayofyear
    
    # 2. Load static features
    sites = pd.read_csv(f"{DRIVE}/sites_with_population.csv")
    sites['lat_cell'] = (sites['latitude'] // 1).astype(int)
    sites['lon_cell'] = (sites['longitude'] // 1).astype(int)
    sites_agg = sites.groupby(['lat_cell', 'lon_cell']).agg({
        'log_population_5km': 'mean', 'dist_to_powerline_km': 'mean'}).reset_index()

    kbdi = pd.read_csv(f"{DRIVE}/kbdi_features.csv")
    fires_meta = pd.read_csv(f"{DRIVE}/fire_signal.csv")
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

    # 4. PLOT WITH ACTUAL GEOGRAPHIC BORDERS
    fig, ax = plt.subplots(figsize=(12, 9))
    
    # Load Natural Earth low-res map (built into geopandas)
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    australia = world[world.name == "Australia"]
    
    # Draw the country outline
    australia.plot(ax=ax, color='#f4f4f4', edgecolor='black', linewidth=1, zorder=1)
    
    # Filter out zero-risk areas so only active risk zones are plotted
    active_risk = forecast_df[forecast_df['risk'] > 0.15]
    
    # Plot the risk cells as squares
    sc = ax.scatter(active_risk['lon_cell'], active_risk['lat_cell'], 
                    c=active_risk['risk'], cmap='YlOrRd', s=80, 
                    marker='s', edgecolor='gray', linewidth=0.3, alpha=0.9, zorder=2)
    
    plt.colorbar(sc, label='Predicted Ignition Probability', fraction=0.046, pad=0.04)
    
    ax.set_xlim(112, 154)
    ax.set_ylim(-44, -10)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('FireSignal Risk Forecast - 15 Nov 2020', fontsize=14, pad=15)
    
    # Clean up the axis borders
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, linestyle='--', alpha=0.3, zorder=0)
    
    st.pyplot(fig)

with tab3:
    st.subheader("Fire Regime Analysis")
    st.markdown("""
    Our research identified differences between human-associated and naturally caused fire activity:
    
    *   **Human-Associated Regime (AUC 0.725):** Fires near infrastructure. Ignition probability is higher, but severity remains lower due to rapid suppression response.
    *   **Remote/Natural Regime (AUC 0.952):** Fires in remote areas. These present significant monitoring and response challenges, and are highly correlated with accumulated drought (measured via the Keetch-Byram Drought Index).
    
    *Implication:* Standard fire indices treat these regimes identically. FireSignal provides an additional information layer to distinguish where decision-makers may need to allocate resources.
    """)
