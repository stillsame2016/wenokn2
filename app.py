import streamlit as st

st.set_page_config(page_title="Redirecting...", layout="centered")

# Option A: Instant redirect via meta refresh (most reliable on Streamlit Cloud)
st.markdown(
    """
    <meta http-equiv="refresh" content="0; url=https://deep-wenokn.streamlit.app/">
    """,
    unsafe_allow_html=True
)

# Fallback content in case the auto-redirect doesn't trigger immediately
st.markdown("## Redirecting to the new app...")

st.markdown(
    "If you are not redirected automatically, please click [here](https://deep-wenokn.streamlit.app/)."
)

# Optional: tiny delay + JavaScript version (more robust in some browsers)
st.markdown(
    """
    <script>
    window.location.replace("https://deep-wenokn.streamlit.app/");
    </script>
    """,
    unsafe_allow_html=True
)
