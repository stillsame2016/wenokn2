import streamlit as st
import streamlit.components.v1 as components

react_app_url = "https://sparcal.sdsc.edu/build/"

# Embed the React app in an iframe
components.iframe(react_app_url, height=400)
