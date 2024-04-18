import streamlit as st
from streamlit_keplergl import keplergl_static
from keplergl import KeplerGl

react_app_url = "https://sparcal.sdsc.edu/build/"

# Embed the React app in an iframe
st.components.iframe(react_app_url, height=400)
