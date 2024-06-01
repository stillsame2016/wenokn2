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
from data_commons import get_variables_for_fips, get_time_series_dataframe_for_fips
from streamlit_extras.chart_container import chart_container


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
    for index, dataset in enumerate(st.session_state.wen_datasets):
        table = st.session_state.wen_datasets[index]
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
                                margin-left: 30px;
                            }
                            </style>
                        """, unsafe_allow_html=True)
            st.write(f"<div class='tableTitle'>Table {index+1}: {dataset.title}</div>", unsafe_allow_html=True)
            ''
            columns = dataset.columns.to_list().remove('Name')
            pivot_table = table.melt(
                ['Name'],
                columns,
                'Year',
                dataset.variable_name,
            )
            pivot_table = pivot_table.dropna()
            # pivot_table['Year'] = pd.to_numeric(pivot_table['Year'])

            min_value = pivot_table['Year'].min()
            max_value = pivot_table['Year'].max()
            
            # min_value, max_value = 1970, 2022
            # from_year, to_year = st.slider(f"Select a time range for Table {index+1}",
            #                         min_value=min_value,
            #                         max_value=max_value,
            #                         value=[min_value, max_value])

            selected_counties = dataset['Name']
            # Filter the data
            filtered_gdp_df = pivot_table[
                (pivot_table['Name'].isin(selected_counties))
                & (pivot_table['Year'] <= max_value) & (min_value <= pivot_table['Year'])
            ]
            
            ''
            st.line_chart(
                filtered_gdp_df,
                x='Year',
                y=dataset.variable_name,
                color='Name',
                height=450
            )
            
            col3, col4 = st.columns([3, 2])

            with col3:
                st.dataframe(pivot_table, hide_index=True, use_container_width=True)
                # st.dataframe(table, width=1100, hide_index=True)

            with col4:
                pass

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
                exec(code)
                df.id = user_input
                st.session_state.wen_datasets.append(df)
                st.session_state.wen_tables.append(df.copy())
                
                # ohio_county_fips = dc.get_places_in(["geoId/39"], 'County')["geoId/39"]
                # df = get_variables_for_fips(ohio_county_fips, ["Count_Person"])
                # df = get_time_series_dataframe_for_fips(ohio_county_fips, "Count_Person")
                # df = get_time_series_dataframe_for_fips(ohio_county_fips, "Count_FloodEvent")
                # df.id = user_input
                # st.session_state.wen_datasets.append(df)
                
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
