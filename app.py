import time
import json
import streamlit as st
from keplergl import keplergl

import pandas as pd
import geopandas as gpd

import google.generativeai as genai

# Setup the datasets in the session for geodataframes
if "datasets" not in st.session_state:
    st.session_state.datasets = []

# Setup the map
options = {"keepExistingConfig": True}
map_config = keplergl(st.session_state.datasets, options=options, config=None, height=400)
time.sleep(0.5)

# Sync datasets and the map
session_data_ids = []
if map_config:
    map_config_json = json.loads(map_config)

    # check if any datasets were deleted
    map_data_ids = [layer["config"]["dataId"] for layer in map_config_json["visState"]["layers"]]
    session_data_ids = [dataset.id for dataset in st.session_state.datasets]
    indices_to_remove = [i for i, dataset in enumerate(st.session_state.datasets) if not dataset.id in map_data_ids]
    for i in reversed(indices_to_remove):
        del st.session_state.datasets[i]

    session_data_ids = [dataset.id for dataset in st.session_state.datasets]
    # st.markdown(session_data_ids)




col1, col2, col3 = st.columns([1, 1, 1])
with col1:
    san_diego_button_clicked = st.button('Add Bart Stops Geo', disabled=("bart-stops-geo" in session_data_ids))
    if san_diego_button_clicked:
        st.session_state.datasets.append(bart_stops_geo_gdf)
        st.rerun()

with col2:
    bart_button_clicked = st.button('Add SF Zip Geo', disabled=("sf-zip-geo" in session_data_ids))
    if bart_button_clicked:
        st.session_state.datasets.append(sf_zip_geo_gdf)
        st.rerun()

with col3:
    h3_button_clicked = st.button('Add H3 Hexagons V2', disabled=("h3-hex-id" in session_data_ids))
    if h3_button_clicked:
        st.session_state.datasets.append(h3_hex_id_df)
        st.rerun()

if map_config:
    st.code(json.dumps(map_config_json, indent=4))






