import time
import json
import streamlit as st
import pandas as pd

from Kepler import my_component
from bart import data
from san_diego import san_diego

st.subheader("Kepler Bidirectional Communication Demo")

if "datasets" not in st.session_state:
    st.session_state.datasets = []

map_config = my_component(json.dumps(st.session_state.datasets), height=400, key="map1")
time.sleep(1)
session_data_ids = []
if map_config:
    map_config_json = json.loads(map_config)
    st.code(json.dumps(map_config_json, indent=4))

    map_data_ids = [layer["dataId"] for layer in map_config_json["layers"]]
    # st.markdown(f"map_data_ids: {map_data_ids}")

    session_data_ids = [dataset['info']['id'] for dataset in st.session_state.datasets]
    # st.markdown(f"session_data_ids: {session_data_ids}")

    indices_to_remove = [i for i, dataset in enumerate(st.session_state.datasets) if
                         not dataset['info']['id'] in map_data_ids]
    # st.markdown(f"indices_to_remove: {indices_to_remove}")

    for i in reversed(indices_to_remove):
        del st.session_state.datasets[i]

    session_data_ids = [dataset['info']['id'] for dataset in st.session_state.datasets]

col1, col2 = st.columns([1, 1])
with col1: 
    san_diego_button_clicked = st.button('Add San Diego Dataset', disabled=("san-diego" in session_data_ids))
    if san_diego_button_clicked:
        st.session_state.datasets.append({
            "info": {"label": "San Diego", "id": "san-diego"},
            "data": san_diego
        })
        st.rerun()

with col2: 
    bart_button_clicked = st.button('Add Bart Dataset', disabled=("bart-stops" in session_data_ids))
    if bart_button_clicked:
        st.session_state.datasets.append({
            "info": {"label": "Bart Stops", "id": "bart-stops"},
            "data": data
        })
        st.rerun()





