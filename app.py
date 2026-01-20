import streamlit as st

NEW_APP_URL = "https://deep-wenokn.streamlit.app"

st.markdown(
    f"""
    <meta http-equiv="refresh" content="0; url={NEW_APP_URL}">
    """,
    unsafe_allow_html=True,
)
