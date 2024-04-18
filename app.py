
import streamlit as st
import time
import streamlit.components.v1 as components

import streamlit as st
import leafmap.kepler as leafmap
from leafmap.common import get_center


# st.set_page_config(layout="wide")
st.markdown("### WEN-OKN: Dive into Data, Never Easier")

m = leafmap.Map(center=[40.4173, -82.9071], zoom=6, height=300)

in_csv = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/hex_data.csv'
config = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/hex_config.json'
m.add_csv(in_csv, layer_name="hex_data", config=config)

m.to_streamlit()
st.markdown(m.config)

iframe_src = "https://open.spotify.com/embed/track/59BweHnnNQc5Y55WO30JuK?utm_source=generator"
components.iframe(iframe_src)
