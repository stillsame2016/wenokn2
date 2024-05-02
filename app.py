
import json
import streamlit as st
import pandas as pd

from Kepler import my_component
from sample_data import data

st.subheader("Kepler Bidirectional Communication Demo")

if "datasets" not in st.session_state:
    st.session_state.datasets = []

map_config = my_component(json.dumps(st.session_state.datasets), height=400, key="map1")

if map_config:
   st.code(json.dumps(json.loads(map_config), indent=4))
