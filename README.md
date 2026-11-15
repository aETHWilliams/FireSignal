# FireSignal

# FireSignal

**Research Question:** Can we use satellite data to anticipate where fire activity will occur next?

## Overview
FireSignal is a spatiotemporal machine learning model designed to forecast the probability of subsequent fire activity across Australia. It acts as a triage map, identifying where attention is needed most urgently.

Standard fire indices rely on weather to indicate when conditions are dangerous. FireSignal demonstrates that harmonised satellite observations contain additional predictive information about where fire activity will occur next, driven by spatial and temporal dependence.

## The Discovery
Weather alone is not enough. The model discovered that previous day fire activity in neighbouring grid cells provides additional predictive information about subsequent fire activity. 

By adding a single feature asking *"Was there a fire next door yesterday?"*, the XGBoost model's predictive accuracy improved:
* **Base Model (Geography + Weather):** AUC 0.810
* **FireSignal (+ Neighbour Fire):** AUC 0.854
* **Statistical Validation:** A paired bootstrap on identical test rows (2,000 iterations) confirmed this +0.044 improvement is significant (95% CI: +0.041 to +0.047, p < 0.001).

## Validation & Generalisation
Stress tested the model to ensure it was not just memorising the 2020 Black Summer data:
* **Forward in time:** Trained on 2019, tested on 2020 (AUC 0.843).
* **Geographic holdout:** Trained on one region, tested on another (Median AUC 0.790).
* **New ignitions only:** Removed ongoing fires (AUC 0.697).

**Limitations:** Tested predicting fire *severity* (size) and it performed poorly (CV R² -0.11). I report this because understanding the limits of satellite derived predictions is just as important as understanding their potential.

## Data Sources
* **Satellite Data:** NASA FIRMS (MODIS + VIIRS) - 486,255 harmonised active fire detections.
* **Environmental Data:** Open-Meteo (Weather), WorldPop (Population density).
* **Infrastructure:** OpenStreetMap (Powerlines).
* **Research Data:** Mulgar (Monash University) for cause attributed fires, Queensland Government for air quality.

## Tech Stack
* **Language:** Python
* **Environment:** Google Colab
* **Machine Learning:** XGBoost, Scikit-learn, SHAP (Explainable AI)
* **Deployment:** Streamlit, PyDeck (Geospatial mapping)

## AI Disclosure
In accordance with NASA Space Apps guidelines, disclose that Qwen3 (8B) running locally via Ollama was used to assist with data cleaning scripts, debugging coordinate geometries, and generating the Streamlit UI code. Core scientific methodology, model selection, and validation testing were led by me.

## Live Application
Test the live dashboard here: https://firesignal-fbkttpxjn8mtghusyuncct.streamlit.app/

***
*We have the data. We have the technology. We need to do better.*
