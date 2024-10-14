import time
import json
import uuid
import streamlit as st
import pandas as pd
import geopandas as gpd
import datacommons_pandas as dc
from keplergl import keplergl
from langchain_groq import ChatGroq

from util import process_data_request, process_regulation_request, process_off_topic_request, process_data_commons_request
from refine_request import get_refined_question
from request_router import get_question_route
from request_plan import get_request_plan
from dataframe_table import render_interface_for_table
from data_commons import get_time_series_dataframe_for_dcid, get_dcid_from_county_name,  get_dcid_from_state_name, get_dcid_from_country_name
from energy_atlas import *
from wenokn_use_energy_atlas import process_wenokn_use_energy_atlas

from streamlit.components.v1 import html

# Setup LLM
Groq_KEY = st.secrets["Groq_KEY"]
Groq_KEY_2 = st.secrets["Groq_KEY_2"]

# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY)
# llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY_2)

llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=Groq_KEY)
llm2 = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=Groq_KEY_2)

# Set the wide layout of the web page
st.set_page_config(layout="wide", page_title="WEN-OKN")

# Set up the title
st.markdown("### &nbsp; WEN-OKN: Dive into Data, Never Easier")
# st.markdown("### &nbsp; Dive into Data, Never Easier")

# Get all query parameters
query_params = st.query_params
init_query = None
if "query" in query_params:
    init_query = query_params["query"]
    # st.write(f"Init Query: {init_query}")

# Set up the datasets in the session for GeoDataframes
if "datasets" not in st.session_state:
    st.session_state.datasets = []

# Add a Chat history object to Streamlit session state
if "chat" not in st.session_state:
    st.session_state.chat = []

# Add datasets for tables
if "wen_datasets" not in st.session_state:
    st.session_state.wen_datasets = []
    st.session_state.wen_tables = []
    st.session_state.table_chat_histories = []
    st.session_state.chart_types = []

# Flag for managing rerun. 
if "rerun" not in st.session_state:
    st.session_state.rerun = False
    
# Add all generated SPARQL queries with the requests to Streamlit session state
if "sparqls" not in st.session_state:
    st.session_state.requests = []
    st.session_state.sparqls = []

if "sample_query" not in st.session_state:
    st.session_state.sample_query = []

# @st.experimental_fragment
@st.fragment(run_every=60*5)
def add_map():
    # st.markdown(f"st.session_state.datasets: {len(st.session_state.datasets)}")
    options = {"keepExistingConfig": True}
    _map_config = keplergl(st.session_state.datasets, options=options, config=None, height=460)
    time.sleep(0.5)

    # Sync datasets saved in the session with the map
    if _map_config:
        map_config_json = json.loads(_map_config)
        # st.code(json.dumps(map_config_json, indent=4))

        # check if any datasets were deleted
        map_data_ids = [layer["config"]["dataId"] for layer in map_config_json["visState"]["layers"]]
        indices_to_remove = [i for i, dataset in enumerate(st.session_state.datasets) if not dataset.id in map_data_ids]    

        deleted = False
        for i in reversed(indices_to_remove):
            # the returnd map config may have several seconds delay 
            if time.time() - st.session_state.datasets[i].time > 3:
                del st.session_state.datasets[i]
                del st.session_state.requests[i]
                del st.session_state.sparqls[i]
                deleted = True
        if deleted:
             st.rerun()
    return _map_config


# Set up CSS for tables
st.markdown("""
            <style>
            .tableTitle {
                font-size: 18pt;
                font-weight: 600;
                color: rgb(49, 51, 63);
                padding: 10px 0px 10px 0px;
            }
            .stDataFrame {
                margin-left: 50px;
            }
            </style>
        """, unsafe_allow_html=True)

# Set up two columns for the map and chat interface
col1, col2 = st.columns([3, 2])

# Show all tables
if st.session_state.wen_datasets:
    for index, pivot_table in enumerate(st.session_state.wen_datasets):
        render_interface_for_table(llm, llm2, index, pivot_table)

# Show all requests and generated SPARQL queries
if len(st.session_state.sparqls) > 0:
    ''
    st.write(f"<div class='tableTitle'>Spatial Requests and SPARQL queries</div>", unsafe_allow_html=True)
    info_container = st.container(height=350)
    with info_container:
        for idx, sparql in enumerate(st.session_state.sparqls):
            if st.session_state.sparqls[idx] != "":
                st.markdown(f"**Request:**  {st.session_state.requests[idx]}")
                st.code(sparql)

# Set up the Kepler map
with col1:
    map_config = add_map()

# Set up the chat interface
with col2:
    # Create a container for the chat messages
    chat_container = st.container(height=355)

    # Show the chat history
    for message in st.session_state.chat:
        with chat_container:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

    # Get user input
    user_input = st.chat_input("What can I help you with?", key="main_chat_input")
    if init_query and len(st.session_state.chat) == 0:
        user_input = init_query
    
    sample_queries = [
        'Find Ross county.', 
        'Find all counties in Ohio State.',
        'Find all counties Scioto River flow through.',
        'Find all counties downstream of Ross county on Scioto River.',
        'Find Ohio River.', 
        'Find all rivers flow through Ross county.',
        'Find flood event counts for all counties downstream of Ross county on Scioto River.',
    ]
    st.markdown(
        """
        <style>
        [data-baseweb="select"] {
            margin-top: -70px;
        }      
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    with open( "./style.css" ) as css:
        st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html=True)
    selected_item = st.selectbox("", 
                                 sample_queries,
                                 index=None,
                                 label_visibility='hidden',
                                 placeholder="Sample Queries",
                                 key='selection_index')
    if selected_item:
        st.session_state.sample_query = [ selected_item ]
        
        
    if user_input:
        with chat_container:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat.append({"role": "user", "content": user_input})
            route = get_question_route(llm, user_input)
            # st.markdown(route)
            if route['request_type'] == 'WEN-KEN database':
                refined_request = get_refined_question(llm, user_input)
                if refined_request['is_request_data']:
                    plan = get_request_plan(llm, refined_request['request'])
                    count_start = len(st.session_state.datasets)
                    # st.code(json.dumps(plan, indent=4))
                    for request in plan['requests']:
                        process_data_request(request, chat_container)
                    count_end = len(st.session_state.datasets)   
                    for idx in range(count_start, count_end):
                        st.session_state.datasets[idx].time = time.time()
                    st.session_state.chat.append({"role": "assistant",
                                                  "content": "Your request has been processed."})
                    st.rerun()
                    # process_data_request(f"{refined_request['request']}", chat_container)
                else:
                    message = refined_request['alternative_answer']
                    st.chat_message("assistant").markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
            elif route['request_type'] == 'NPDES regulations':
                message = process_regulation_request(llm, user_input, chat_container)
                st.chat_message("assistant").markdown(message)
                st.session_state.chat.append({"role": "assistant", "content": message})
                st.rerun()
            elif route['request_type'] == 'Data Commons':
                code = process_data_commons_request(llm, user_input, st.session_state.datasets)
                # st.code(code)
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        try:
                            exec(code)
                            df.id = user_input
                            st.session_state.wen_datasets.append(df)
                            st.session_state.wen_tables.append(df.copy())
                            st.session_state.table_chat_histories.append([])
                            st.session_state.chart_types.append("bar_chart")
                            message = f"""
                                    Your request has been processed. {df.shape[0]} { "rows are" if df.shape[0] > 1 else "row is"}
                                    found and displayed.
                                    """
                        except Exception as e:
                            message = f"""
                                       {code} 
                                       {str(e)}
                                       """               
                            # message = f"""We are not able to process your request. Please refine your 
                            #               request and try it again. \n\nError: {str(e)}"""
                        st.markdown(message)
                        st.session_state.chat.append({"role": "assistant", "content": message})
                        st.rerun()
            elif route['request_type'] == 'US Energy Atlas':
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        try:
                            code = process_energy_atlas_request(llm, user_input, st.session_state.datasets)
                            exec(code)
                            # st.code(code)
                            if gdf.shape[0] > 0:
                                if hasattr(gdf, 'answer'):
                                    message = gdf.answer
                                else:
                                    gdf.label = gdf.title
                                    gdf.id = str(uuid.uuid4())[:8]
                                    gdf.time = time.time()
                                    st.session_state.requests.append(user_input)
                                    st.session_state.sparqls.append("")
                                    st.session_state.datasets.append(gdf)
                                    st.session_state.rerun = True
                                    message = f"""
                                                Your request has been processed. {gdf.shape[0]} 
                                                { "items are" if gdf.shape[0] > 1 else "item is"}
                                                loaded on the map.
                                                """
                            else:
                                message = f"""
                                            Your request has been processed. Nothing was found.
                                            Please refine your request and try again if you think
                                            this is a mistake.
                                            """
                        except Exception as e:
                            # message = f"""
                            #            {code} 
                            #            {str(e)}
                            #            """  
                            message = f"""We are not able to process your request. Please refine your 
                                              request and try it again. \n\nError: {str(e)}"""
                    st.markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
            elif route['request_type'] == "WEN-KEN database use Energy Atlas":
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        code = process_wenokn_use_energy_atlas(llm, user_input)
                        # st.code(code)
                        exec(code)
                        st.markdown(f"Loaded data from Energy Atlas and converted the request to: {converted_request}")
                    
                    process_data_request(converted_request, chat_container)
                    st.session_state.datasets[-1].label = user_input
                    st.session_state.requests[-1] = user_input
                    message = "Your request has been processed."
                    
                    st.markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
                    st.rerun()
            else:
                message = process_off_topic_request(llm, user_input, chat_container)
                st.chat_message("assistant").markdown(message)
                st.session_state.chat.append({"role": "assistant", "content": message})
                st.rerun()

if st.session_state.rerun:
    st.session_state.rerun = False
    st.rerun()


st.markdown("")
st.markdown("")

# Initialize the session state variables
if 'selectbox_key' not in st.session_state:
    st.session_state.selectbox_key = 0
if 'selected_option' not in st.session_state:
    st.session_state.selected_option = None

def clear_selection():
    st.session_state.selectbox_key += 1
    st.session_state.selected_option = None

def on_change():
    st.session_state.selected_option = st.session_state.temp_select

options = ["Option 1", "Option 2", "Option 3"]

# Create a selectbox with a dynamic key
option = st.selectbox(
    "Choose an option",
    options,
    key=f"my_selectbox_{st.session_state.selectbox_key}",
    index=None if st.session_state.selected_option is None else options.index(st.session_state.selected_option),
    on_change=on_change,
    key="temp_select"
)

# Create a button to clear the selectbox
st.button("Clear Selection", on_click=clear_selection)

# Display the current selection
st.write("You selected:", st.session_state.selected_option if st.session_state.selected_option else "No selection")

if st.session_state.sample_query:
    # st.markdown(st.session_state.sample_query)
    js_code = f"""
            <script>
            const doc = window.parent.document;
            const chatInput = doc.querySelector('.stChatInput textarea');
            chatInput.focus();
            function autoResizeTextarea() {{
                // chatInput.value = '{st.session_state.sample_query[0]}';   
                chatInput.style.height = 'auto';
                chatInput.style.height = chatInput.scrollHeight + 'px';
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, "{st.session_state.sample_query[0]} ");
                const event = new Event('input', {{ bubbles: true }});
                chatInput.dispatchEvent(event);
            }}
            setTimeout(autoResizeTextarea, 100);
            </script>
            """
    html(js_code)


# if map_config:
#     map_config_json = json.loads(map_config)
#     st.code(json.dumps(map_config_json, indent=4))
