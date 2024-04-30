
import json
import streamlit as st
import pandas as pd

from Kepler import my_component

st.subheader("Kepler Bi-Direction Connection Dev")

map_config = my_component("""
   [ 
      { 
         info: {label: 'Bart Stops Geo', id: 'bart-stops-geo'}, 
         data: { test: 123 }
      }
   ]
""")
st.code(json.loads(map_config))
