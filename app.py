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
    
    # 1. Generate the full grid
    lat_grid = np.arange(-44, -9, 1)
    lon_grid = np.arange(113, 155, 1)
    grid_cells = [{'lat_cell': float(lat), 'lon_cell': float(lon)} for lat in lat_grid for lon in lon_grid]
    forecast_df = pd.DataFrame(grid_cells)
    forecast_df['doy'] = pd.to_datetime('2020-11-15').dayofyear
    
    # 2. Load features
    sites = pd.read_csv(f"{DRIVE}/sites_with_population.csv")
    sites['lat_cell'] = (sites['latitude'] // 1).astype(int)
    sites['lon_cell'] = (sites['longitude'] // 1).astype(int)
    sites_agg = sites.groupby(['lat_cell', 'lon_cell']).agg({
        'log_population_5km': 'mean', 'dist_to_powerline_km': 'mean'}).reset_index()

    kbdi = pd.read_csv(f"{DRIVE}/kbdi_features.csv")
    fires_meta = pd.read_csv(f"{DRIVE}/fires_signal.csv")
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

    # 4. PLOT WITH GEOGRAPHIC CLIPPING
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='white')
    
    import urllib.request, json
    import geopandas as gpd
    from shapely.geometry import Point
    
    try:
        # Fetch Australia shapefile
        url = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_admin_0_countries.geojson"
        with urllib.request.urlopen(url) as response:
            world_data = json.load(response)
        world = gpd.GeoDataFrame.from_features(world_data['features'])
        australia = world[world['NAME'] == "Australia"]
        
        # Draw the landmass
        australia.plot(ax=ax, color='#f0f0f0', edgecolor='#333333', linewidth=1.5, zorder=1)
        
        # --- THE FIX: Only plot grid cells that fall ON LAND ---
        # Create a geometry column for the forecast grid
        geometry = [Point(xy) for xy in zip(forecast_df['lon_cell'], forecast_df['lat_cell'])]
        forecast_gdf = gpd.GeoDataFrame(forecast_df, geometry=geometry, crs="EPSG:4326")
        
        # Clip the grid to the Australia shapefile
        clipped_grid = gpd.clip(forecast_gdf, australia)
        
    except Exception as e:
        st.warning(f"Could not load map background: {e}")
        clipped_grid = forecast_df # Fallback

    # Filter out low-risk areas
    active_risk = clipped_grid[clipped_grid['risk'] > 0.15]
    
    # Plot the clipped risk cells
    sc = ax.scatter(active_risk['lon_cell'], active_risk['lat_cell'], 
                    c=active_risk['risk'], cmap='magma_r', s=100, 
                    marker='s', edgecolor='black', linewidth=0.2, alpha=0.85, zorder=2)
    
    cbar = plt.colorbar(sc, label='Ignition Probability', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlim(112, 154)
    ax.set_ylim(-44, -10)
    ax.set_title('FireSignal Ignition Probability - 15 Nov 2020', fontsize=16, pad=15, fontweight='bold')
    
    # Remove the box border for a cleaner look
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    st.pyplot(fig)

with tab3:
    st.subheader("Fire Regime Analysis")
    st.markdown("""
    Our research identified differences between human-associated and naturally caused fire activity:
    
    *   **Human-Associated Regime (AUC 0.725):** Fires near infrastructure. Ignition probability is higher, but severity remains lower due to rapid suppression response.
    *   **Remote/Natural Regime (AUC 0.952):** Fires in remote areas. These present significant monitoring and response challenges, and are highly correlated with accumulated drought (measured via the Keetch-Byram Drought Index).
    
    *Implication:* Standard fire indices treat these regimes identically. FireSignal provides an additional information layer to distinguish where decision-makers may need to allocate resources.
    """)
