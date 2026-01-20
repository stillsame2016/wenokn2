import streamlit as st

# Define the destination URL
new_app_url = "https://deep-wenokn.streamlit.app"

# Display a message to the user in case the redirect takes a moment
st.title("Redirecting...")
st.write(f"This app is deprecated. Moving you to the new version at {new_app_url}")

# Inject JavaScript to perform the redirect
st.components.v1.html(
    f"""
    <script>
        window.parent.location.href = "{new_app_url}";
    </script>
    """,
    height=0,
)
