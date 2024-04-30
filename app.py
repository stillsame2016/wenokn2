import streamlit as st
import pandas as pd

from Kepler import my_component

st.subheader("Kepler Bi-Direction Connection Dev")

# Create an instance of our component with a constant `name` arg, and
# print its output value.
num_clicks = my_component("""
   [ 
      { 
         info: {label: 'Bart Stops Geo', id: 'bart-stops-geo'}, 
         data: { test: 123 }
      }
   ]
""")
st.code(num_clicks)
