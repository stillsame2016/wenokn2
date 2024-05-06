import time
import json
import streamlit as st
from keplergl import keplergl

import pandas as pd
import geopandas as gpd

import google.generativeai as genai

# Set the wide layout of the web page
st.set_page_config(layout="wide", page_title="WEN-OKN")

# Setup the title
st.markdown("### WEN-OKN: Dive into Data, Never Easier")

# Setup the datasets in the session for geodataframes
if "datasets" not in st.session_state:
    st.session_state.datasets = []

# Setup LLM model
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-pro')

# The safty setting for Gemini-Pro 
safe = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
]

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = model.start_chat(history=[])

if "wen_datasets" not in st.session_state:
    st.session_state.wen_datasets = []

# Add all generated sparqls to Streamlit session state
if "sparqls" not in st.session_state:
    st.session_state.requests = []
    st.session_state.sparqls = []

def get_column_name_parts(column_name):
    return re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)', column_name)
    
def df_to_gdf(df):
  column_names = df.columns.tolist()
  geometry_column_names = [ x for x in column_names if x.endswith('Geometry')]
  df['geometry'] = df[geometry_column_names[0]].apply(wkt.loads)
  gdf = gpd.GeoDataFrame(df, geometry='geometry')
  gdf.drop(columns=[geometry_column_names[0]], inplace=True)
  
  column_name_parts = get_column_name_parts(column_names[0])
  column_name_parts.pop()
  gdf.attrs['data_name'] = " ".join(column_name_parts).capitalize()
  
  for column_name in column_names:
    tmp_column_name_parts = get_column_name_parts(column_name)
    tmp_name = tmp_column_name_parts.pop()  
    tmp_data_name = " ".join(column_name_parts).capitalize()
    if gdf.attrs['data_name'] == tmp_data_name:
      gdf.rename(columns={column_name: tmp_name}, inplace=True)
  # if tmp_data_name == gdf.attrs['data_name']:
  #     gdf.rename(columns={column_name: name}, inplace=True)
  return gdf

# Function to add a new message to the chat
def add_message(sender, message, processing=False):
    with chat_container:
      if processing:
        with st.chat_message("assistant"):
          with st.spinner(f"""We're currently processing your request:
                                    **{message}{'' if message.endswith('.') else '.'}**
                              Depending on the complexity of the query and the volume of data, 
                              this may take a moment. We appreciate your patience."""):
            max_tries = 5
            tried = 0
            gdf_empty = False
            while tried < max_tries:
              try:
                  response = requests.get(f"https://sparcal.sdsc.edu/staging-api/v1/Utility/wenokn_llama3?query_text={message}")
                  data = response.text.replace('\\n', '\n').replace('\\"', '"').replace('\\t', ' ')
                  if data.startswith("\"```sparql"):
                    start_index = data.find("```sparql") + len("```sparql")
                    end_index = data.find("```", start_index)
                    sparql_query = data[start_index:end_index].strip()
                  elif data.startswith("\"```code"):
                    start_index = data.find("```code") + len("```code")
                    end_index = data.find("```", start_index)
                    sparql_query = data[start_index:end_index].strip()
                  elif data.startswith("\"```"):
                    start_index = data.find("```") + len("```")
                    end_index = data.find("```", start_index)
                    sparql_query = data[start_index:end_index].strip()
                  elif data.startswith('"') and data.endswith('"'):
                    # Remove leading and trailing double quotes
                    sparql_query = data[1:-1]
                  else:
                    sparql_query = data
                  sparql_query = sparql_query.replace("\n\n\n", "\n\n")
                  
                  st.markdown(
                      """
                      <style>
                      .st-code > pre {
                          font-size: 0.4em; 
                      }
                      </style>
                      """,
                      unsafe_allow_html=True
                    )
                  st.code(sparql_query)
                  
                  endpoint = "http://132.249.238.155/repositories/wenokn_ohio_all"
                  df = sparql_dataframe.get(endpoint, sparql_query)   
                  
                  gdf = df_to_gdf(df)
                  if gdf.shape[0] == 0:
                    # double check
                    if not gdf_empty:
                      gdf_empty = True
                      tried += 1
                      continue

                  tried = max_tries + 10
                  st.session_state.requests.append(message)
                  st.session_state.sparqls.append(sparql_query)
                  st.session_state.wen_datasets.append(gdf)  
                  st.rerun()
              except Exception as e:
                st.markdown(f"Encounter an error: {str(e)}. Try again...")
                traceback.print_exc()
                tried += 1               
            if tried == max_tries:
              st.markdown("We are not able to process your request at this moment. You can try it again now or later.") 
      else: 
         st.chat_message(sender).write(message)


col1, col2 = st.columns([6, 4])

info_container = st.container(height=350)
with info_container:
  for idx, sparql in enumerate(st.session_state.sparqls): 
    st.markdown(f"**Request:**  {st.session_state.requests[idx]}")
    st.code(sparql)

with col1:  
  # Setup the map
  options = {"keepExistingConfig": True}
  map_config = keplergl(st.session_state.datasets, options=options, config=None, height=410)
  time.sleep(0.5)
  
  # Sync datasets and the map
  session_data_ids = []
  if map_config:
      map_config_json = json.loads(map_config)
  
      # check if any datasets were deleted
      map_data_ids = [layer["config"]["dataId"] for layer in map_config_json["visState"]["layers"]]
      session_data_ids = [dataset.id for dataset in st.session_state.datasets]
      indices_to_remove = [i for i, dataset in enumerate(st.session_state.datasets) if not dataset.id in map_data_ids]
      for i in reversed(indices_to_remove):
          del st.session_state.datasets[i]
  
      session_data_ids = [dataset.id for dataset in st.session_state.datasets]
      # st.markdown(session_data_ids)

with col2:
    
    # Create a container for the chat messages
    chat_container = st.container(height=355)
    
    # Get user input
    user_input = st.chat_input("What can I help you with?")

    if user_input:
        # Add user message to the chat
        add_message("User", user_input)


if map_config:
    st.code(json.dumps(map_config_json, indent=4))






