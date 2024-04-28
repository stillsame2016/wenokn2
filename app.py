import streamlit as st
import pandas as pd

# import streamlit.components.v1 as components

# react_app_url = "https://sparcal.sdsc.edu/build/"

# # Embed the React app in an iframe
# components.iframe(react_app_url, height=400)

import geemap.kepler as geemap
import time

m = geemap.Map(center=[40, -100], zoom=2, height=600, widescreen=False)
m.to_streamlit(width=600, height=400)

df = pd.DataFrame(
    {
        "City": ["San Francisco", "San Jose", "Palo Alto"],
        "Latitude": [37.77, 37.33, 37.44],
        "Longitude": [-122.43, -121.89, -122.14],
    }
)

m.add_data(
    data=df, name="cities"
) 



