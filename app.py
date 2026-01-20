import streamlit as st

NEW_APP_URL = "https://deep-wenokn.streamlit.app"

st.markdown(
    f"""
    <script>
        window.location.replace("{NEW_APP_URL}");
    </script>
    """,
    unsafe_allow_html=True,
)
