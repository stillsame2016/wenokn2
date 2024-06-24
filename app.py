import time
import json
import streamlit as st
import datacommons_pandas as dc
from keplergl import keplergl
from langchain_groq import ChatGroq

from util import process_data_request, process_regulation_request, process_off_topic_request,  process_data_commons_request
from refine_request import get_refined_question
from request_router import get_question_route
from request_plan import get_request_plan
from dataframe_table import render_interface_for_table
from data_commons import get_time_series_dataframe_for_dcid, get_dcid_from_county_name,  get_dcid_from_state_name, get_dcid_from_country_name


# Setup LLM
Groq_KEY = st.secrets["Groq_KEY"]
Groq_KEY_2 = st.secrets["Groq_KEY_2"]

llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY)
llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY_2)

# Set the wide layout of the web page
st.set_page_config(layout="wide", page_title="WEN-OKN")

# Set up the title
st.markdown("### &nbsp; WEN-OKN: Dive into Data, Never Easier")

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


@st.experimental_fragment
def add_map():
    # st.markdown(f"st.session_state.datasets: {len(st.session_state.datasets)}")
    options = {"keepExistingConfig": True}
    _map_config = keplergl(st.session_state.datasets, options=options, config=None, height=410)
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
    user_input = st.chat_input("What can I help you with?")

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
                            # message = f"""
                            #            {code} 
                            #            {str(e)}
                            #            """               
                            message = f"""We are not able to process your request. Please refine your 
                                          request and try it again. \n\nError: {str(e)}"""
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

# if map_config:
#     map_config_json = json.loads(map_config)
#     st.code(json.dumps(map_config_json, indent=4))
