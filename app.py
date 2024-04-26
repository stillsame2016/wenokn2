# import streamlit as st
# import streamlit.components.v1 as components

# react_app_url = "https://sparcal.sdsc.edu/build/"

# # Embed the React app in an iframe
# components.iframe(react_app_url, height=400)

import geemap.kepler as geemap
import time

m = geemap.Map(center=[40, -100], zoom=2, height=600, widescreen=False)
m.to_streamlit(width=800, height=600)

time.sleep(10000)

polygons = "https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_states.json"
m.add_geojson(polygons, layer_name="Countries")

