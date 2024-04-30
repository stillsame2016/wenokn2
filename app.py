
import json
import streamlit as st
import pandas as pd

from Kepler import my_component

st.subheader("Kepler Bidirectional Connection Demo")

datasets = []
map_config = my_component(json.dumps(datasets), key="map1")

if map_config:
   st.code(json.dumps(json.loads(map_config), indent=4))
