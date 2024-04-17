"""Run 'streamlit run app.py' in the terminal to start the app.
"""
import streamlit as st
import time
import streamlit.components.v1 as components

# st.set_page_config(layout="wide")

st.markdown("### WEN-OKN: Dive into Data, Never Easier")

# "# leafmap streamlit demo"
# st.markdown('Source code: <https://github.com/giswqs/leafmap-streamlit/blob/master/app.py>')

# "## Create a 3D map using Kepler.gl"
# with st.echo():

import streamlit as st
import leafmap.kepler as leafmap

from leafmap.common import get_center

m = leafmap.Map(center=[40.4173, -82.9071], zoom=6, height=400)

# in_csv = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/hex_data.csv'
# config = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/hex_config.json'
# m.add_csv(in_csv, layer_name="hex_data", config=config)

m.to_streamlit()
st.markdown(m.config)

iframe_src = "https://open.spotify.com/embed/track/59BweHnnNQc5Y55WO30JuK?utm_source=generator"
components.iframe(iframe_src)

# "## Create a heat map"
# with st.echo():
#     import leafmap.foliumap as leafmap

#     filepath = "https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/us_cities.csv"
#     m = leafmap.Map(tiles='stamentoner')
#     m.add_heatmap(filepath, latitude="latitude", longitude='longitude', value="pop_max", name="Heat map", radius=20)
#     m.to_streamlit(width=700, height=500, add_layer_control=True)

# "## Load a GeoJSON file"
# with st.echo():
#     m = leafmap.Map(center=[0, 0], zoom=2)
#     in_geojson = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/cable-geo.geojson'
#     m.add_geojson(in_geojson, layer_name="Cable lines")
#     m.to_streamlit()

# "## Add a colorbar"
# with st.echo():
#     m = leafmap.Map()
#     m.add_basemap('USGS 3DEP Elevation')
#     colors = ['006633', 'E5FFCC', '662A00', 'D8D8D8', 'F5F5F5']
#     vmin = 0
#     vmax = 4000
#     m.add_colorbar(colors=colors, vmin=vmin, vmax=vmax)
#     m.to_streamlit()

# "## Change basemaps"
# with st.echo():
#     m = leafmap.Map()
#     m.add_basemap("Esri.NatGeoWorldMap")
#     m.to_streamlit()

