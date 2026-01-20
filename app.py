import streamlit as st

st.markdown(
    """
    <script>
    window.location.replace("https://deep-wenokn.streamlit.app/");
    </script>
    <noscript>
        <meta http-equiv="refresh" content="0; url=https://deep-wenokn.streamlit.app/">
        <p>Redirecting to the new app... If nothing happens, <a href="https://deep-wenokn.streamlit.app/">click here</a>.</p>
    </noscript>
    """,
    unsafe_allow_html=True
)
