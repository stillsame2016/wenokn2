import time
import json
import streamlit as st
from keplergl import keplergl
from util import process_data_request, process_regulation_request, process_off_topic_request, process_data_commons_request
from langchain_groq import ChatGroq
from refine_request import get_refined_question
from request_router import get_question_route
from request_plan import get_request_plan

import pandas as pd
import datacommons_pandas as dc
from data_commons import get_time_series_dataframe_for_fips, get_fips_from_county_name, get_fips_from_state_name
from streamlit_extras.chart_container import chart_container
from streamlit_extras.stylable_container import stylable_container


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

# Add datasets created by queries to the session and display by the map
if "wen_datasets" not in st.session_state:
    st.session_state.wen_datasets = []
    st.session_state.wen_tables = []
    st.session_state.table_chat_histories = []
    

# Add all generated SPARQL queries with the requests to Streamlit session state
if "sparqls" not in st.session_state:
    st.session_state.requests = []
    st.session_state.sparqls = []

# Set up two columns for the map and chat interface
col1, col2 = st.columns([3, 2])


@st.experimental_fragment
def add_map():
    options = {"keepExistingConfig": True}
    _map_config = keplergl(st.session_state.datasets, options=options, config=None, height=410)
    time.sleep(0.5)

    # Sync datasets saved in the session with the map
    if _map_config:
        map_config_json = json.loads(_map_config)

        # check if any datasets were deleted
        map_data_ids = [layer["config"]["dataId"] for layer in map_config_json["visState"]["layers"]]
        indices_to_remove = [i for i, dataset in enumerate(st.session_state.datasets) if not dataset.id in map_data_ids]
        for i in reversed(indices_to_remove):
            del st.session_state.datasets[i]

    return _map_config

# Process tables
if st.session_state.wen_datasets:
    for index, pivot_table in enumerate(st.session_state.wen_datasets):
        buffered_table = st.session_state.wen_tables[index]
        with st.container(): 
            ''
            st.markdown("""
                            <style>
                            .tableTitle {
                                font-size: 18pt;
                                font-weight: 600;
                                color: rgb(49, 51, 63);
                                padding: 0px 0px 10px 0px;
                            }
                            
                            .stSlider [data-baseweb=slider]{
                                width: 90%;
                                margin: 0px 50px 0px 30px;
                            }

                            .stDataFrame {
                                margin-left: 50px;
                            }
                            </style>
                        """, unsafe_allow_html=True)
            st.write(f"<div class='tableTitle'>Table {index+1}: {pivot_table.title}</div>", unsafe_allow_html=True)
            ''
            min_value = pivot_table['Date'].min()
            max_value = pivot_table['Date'].max()
            
            selected_counties = pivot_table['Name']
            # Filter the data
            filtered_pivot_table = pivot_table[
                (pivot_table['Name'].isin(selected_counties))
                & (pivot_table['Date'] <= max_value) & (min_value <= pivot_table['Date'])
            ]
            
            ''
            st.bar_chart(
                filtered_pivot_table,
                x='Date',
                y=pivot_table.variable_name,
                color='Name',
                height=450
            )
            
            col3, pad, col4 = st.columns([30, 3, 20])
            with col3:
                st.dataframe(buffered_table, hide_index=True, use_container_width=True)
            with col4:
                table_chat_container = st.container(height=340)
                user_input_for_table = st.chat_input(f"What can I help you with Table {index+1}?")
                if user_input_for_table:
                    with table_chat_container:

                        # Show the chat history
                        for message in st.session_state.table_chat_histories[index]:
                            with st.chat_message(message['role']):
                                st.markdown(message['content'])
                        
                        st.chat_message("user").markdown(user_input_for_table)
                        st.session_state.table_chat_histories[index].append({"role": "user", "content": user_input_for_table})

                        response = f"echo: {user_input_for_table}"
                        st.chat_message("assistant").markdown(response)
                        st.session_state.table_chat_histories[index].append({"role": "assistant", "content": response})
                

# Show all requests and generated SPARQL queries
if len(st.session_state.sparqls) > 0:
    info_container = st.container(height=350)
    with info_container:
        for idx, sparql in enumerate(st.session_state.sparqls):
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
            if route['request_type'] == 'WEN-KEN database':
                refined_request = get_refined_question(llm, user_input)
                if refined_request['is_request_data']:
                    plan = get_request_plan(llm, refined_request['request'])
                    # st.code(json.dumps(plan, indent=4))
                    for request in plan['requests']:
                        process_data_request(request, chat_container)
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
                code = process_data_commons_request(llm, user_input, chat_container)
                st.code(code)
                try:
                    exec(code)
                    df.id = user_input
                    st.session_state.wen_datasets.append(df)
                    st.session_state.wen_tables.append(df.copy())
                    st.session_state.table_chat_histories.append([])
                except Exception as e:
                    st.markdown(str(e))                
                message = f"""
                            Your request has been processed. {df.shape[0]} { "rows are" if df.shape[0] > 1 else "row is"}
                            found and displayed.
                            """
                st.chat_message("assistant").markdown(message)
                st.session_state.chat.append({"role": "assistant", "content": message})
                st.rerun()
            else:
                message = process_off_topic_request(llm, user_input, chat_container)
                st.chat_message("assistant").markdown(message)
                st.session_state.chat.append({"role": "assistant", "content": message})
                st.rerun()
