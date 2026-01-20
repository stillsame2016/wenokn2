import streamlit as st

st.markdown(
    """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Redirecting...</title>
        <script>
            window.location.replace("https://deep-wenokn.streamlit.app/");
        </script>
    </head>
    <body>
        <p>Redirecting to the new app...</p>
        <p>If it doesn't happen automatically, <a href="https://deep-wenokn.streamlit.app/">click here</a>.</p>
    </body>
    </html>
    """,
    unsafe_allow_html=True
)
