import streamlit as st
import streamlit.components.v1 as components

NEW_APP_URL = "https://deep-wenokn.streamlit.app"

components.html(
    f"""
    <script>
        window.location.href = "{NEW_APP_URL}";
    </script>
    """,
    height=0,
)
st.stop()
