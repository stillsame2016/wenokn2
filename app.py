
import json
import streamlit as st
import pandas as pd

from Kepler import my_component
from bart import data

st.subheader("Kepler Bidirectional Communication Demo")

if "datasets" not in st.session_state:
    st.session_state.datasets = []

map_config = my_component(json.dumps(st.session_state.datasets), height=400, key="map1")
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

    # st.markdown(f"final: {len(st.session_state.datasets)}")

button_value = st.button('Add Dataset')
if button_value and len(st.session_state.datasets) == 0:
    st.session_state.datasets.append({
        "info": {"label": "Bart Stops Geo", "id": "bart-stops-geo"},
        "data": data
    })
    st.rerun()

