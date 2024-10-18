import time
import json
import uuid
import requests
import streamlit as st
import pandas as pd
import geopandas as gpd
import datacommons_pandas as dc
from keplergl import keplergl
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from util import process_data_request, process_regulation_request, process_off_topic_request, process_data_commons_request, strip_code, normalize_query_plan
from refine_request import get_refined_question
from request_router import get_question_route
from request_plan import get_request_plan
from dataframe_table import render_interface_for_table
from data_commons import get_time_series_dataframe_for_dcid, get_dcid_from_county_name,  get_dcid_from_state_name, get_dcid_from_country_name, get_variables_for_dcid
from energy_atlas import *
from wenokn_use_energy_atlas import process_wenokn_use_energy_atlas

from streamlit.components.v1 import html

# Setup LLM
Groq_KEY = st.secrets["Groq_KEY"]
Groq_KEY_2 = st.secrets["Groq_KEY_2"]
OpenAI_KEY = st.secrets["OpenAI_KEY"]

# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY)
# llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY_2)

llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=Groq_KEY)
llm2 = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=Groq_KEY_2)

llm = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=5000, api_key=OpenAI_KEY)
llm2 = ChatOpenAI(model="gpt-4o", temperature=0, max_tokens=5000, api_key=OpenAI_KEY)

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
    st.session_state.sample_query = None

if "selection_index" not in st.session_state:
    st.session_state.selection_index = None

if "delete_history" not in st.session_state:
    st.session_state.delete_history = []

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

def ordinal(n):
    suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
    if n % 100 in [11, 12, 13]:  # Special case for 11th, 12th, 13th
        return f"{n}th"
    return f"{n}{suffix[n % 10]}"

def execute_query(user_input, chat_container):
    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/plan?query={user_input}")
    query_plan_text = None
    message = None
    if response.status_code == 200:
        query_plan = json.loads(response.text)
        query_plan = normalize_query_plan(query_plan)
        st.code(json.dumps(query_plan, indent=4))
        if len(query_plan) > 1:
            # show the query plan
            query_plan_text = "We use the following query plan for your request:\n"
            for i, query in enumerate(query_plan, 1):
                query_plan_text += f"{i}. {query['request']}\n"
            st.markdown(query_plan_text)
            
            count_start = len(st.session_state.datasets)
            for i, query in enumerate(query_plan, 1):
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(f"Processing the {ordinal(i)} query in the query plan: **{query['request']}**")
                        if query["data_source"] == "WEN-OKN Database":
                            process_data_request(query["request"], chat_container)
                        elif query["data_source"] == "Data Commons":
                            code = process_data_commons_request(llm, user_input, st.session_state.datasets)
                            code = strip_code(code)
                            # st.code(code)
                            globals_dict = {
                                'st': st,
                                "get_variables_for_dcid": get_variables_for_dcid,
                                "get_time_series_dataframe_for_dcid": get_time_series_dataframe_for_dcid,
                                "get_dcid_from_county_name": get_dcid_from_county_name,
                                "get_dcid_from_state_name": get_dcid_from_state_name,
                                "get_dcid_from_country_name": get_dcid_from_country_name
                            }
                            
                            exec(code, globals_dict)
                            df = globals_dict['df']    
                            df.id = user_input
                            st.session_state.wen_datasets.append(df)
                            st.session_state.wen_tables.append(df.copy())
                            st.session_state.table_chat_histories.append([])
                            st.session_state.chart_types.append("bar_chart")
                            message = f"""
                                    Your request has been processed. {df.shape[0]} { "rows are" if df.shape[0] > 1 else "row is"}
                                    found and displayed.
                                    """
                        elif query["data_source"] == "Energy Atlas":
                            code = process_energy_atlas_request(llm, query["request"], st.session_state.datasets)
                            code = strip_code(code)
                            globals_dict = {
                                'st': st,
                                'load_coal_mines': load_coal_mines,
                                'load_coal_power_plants': load_coal_power_plants,
                                'load_wind_power_plants': load_wind_power_plants,
                                'load_renewable_diesel_fuel_and_other_biofuel_plants': load_renewable_diesel_fuel_and_other_biofuel_plants,
                                'load_battery_storage_plants': load_battery_storage_plants,
                                'load_geothermal_power_plants': load_geothermal_power_plants,
                                'load_hydro_pumped_storage_power_plants': load_hydro_pumped_storage_power_plants,
                                'load_natural_gas_power_plants': load_natural_gas_power_plants,
                                'load_nuclear_power_plants': load_nuclear_power_plants,
                                'load_petroleum_power_plants': load_petroleum_power_plants,
                                'load_solar_power_plants': load_solar_power_plants,
                                'load_biodiesel_plants': load_biodiesel_plants
                            }
                            exec(code, globals_dict)
                            gdf = globals_dict['gdf']
                            if gdf.shape[0] > 0:
                                if hasattr(gdf, 'answer'):
                                    message = gdf.answer
                                else:
                                    gdf.label = gdf.title
                                    gdf.id = str(uuid.uuid4())[:8]
                                    gdf.time = time.time()
                                    st.session_state.requests.append(query["request"])
                                    st.session_state.sparqls.append("")
                                    st.session_state.datasets.append(gdf)
                                    # st.session_state.rerun = True
                                    message = f"""
                                                Your request has been processed. {gdf.shape[0]} 
                                                { "items are" if gdf.shape[0] > 1 else "item is"}
                                                loaded on the map.
                                                """
                            else:
                                raise ValueError(f'The request {query["request"]} has been processed. Nothing was found.')
            count_end = len(st.session_state.datasets)   
            for idx in range(count_start, count_end):
                st.session_state.datasets[idx].time = time.time()
            st.session_state.rerun = True
    return query_plan_text, message

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
        ######## County ########
        'Find Ross County.', 
        'Find all counties in Ohio State.', 
        'Find all counties the Scioto River flows through.',
        'Find all counties downstream of Ross County on the Scioto River.',  
        'Find all counties both the Ohio River and the Muskingum River flow through.',  
        'Show all counties downstream of the coal mine with the name Century Mine along Ohio River.',

        ######## River ########
        'Find the Ohio River.', 
        'Find all rivers that flow through Ross County.', 
        'What rivers flow through Dane County in Wisconsin?', 

        ######## Gages ########
        'What stream gages are on the Yahara River in Madison, WI?',  

        ######## Dam ########
        'Find all dams on the Ohio River.', 
        'Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river',
        
        ######## Data Commons ########
        'Find the populations for all counties in Ohio State.', 
        'Find the median individual income for Ross County and Scioto County.', 
        'Find the number of people employed in all counties the Scioto River flows through.', 

        ######## Energy Atlas ########
        'Show all coal mines in Ohio State.', 
        'Find all coal mines along the Ohio River.', 
        'Load all wind power plants with total megawatt capacity greater than 100 in California.' 
        
        # 'Find flood event counts for all counties downstream of Ross county on Scioto River.',
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
    selected_item = st.selectbox(" ", 
                                 sample_queries,
                                 index=None,
                                 label_visibility='hidden',
                                 placeholder="Sample Queries",
                                 key='selection_index')
    if selected_item:
        st.session_state.sample_query = selected_item  
        
    if user_input:
        st.session_state.sample_query = None
        with chat_container:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat.append({"role": "user", "content": user_input})
            route = get_question_route(llm, user_input)
            # st.markdown(route)
            # time.sleep(20)
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
                            message = None                           
                            try:
                                query_plan_text, message = execute_query(user_input, chat_container)
                            except Exception as error:
                                st.code(f"{str(error)}")
                                pass

                            if message is None:
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
                            try:
                                query_plan_text, message = execute_query(user_input, chat_container)
                            except:
                                pass
             
                    st.markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
            elif route['request_type'] == "WEN-KEN database use Energy Atlas":
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        code = process_wenokn_use_energy_atlas(llm, user_input)
                        code = strip_code(code)
                        # st.code(code)
                        # time.sleep(20)
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

if st.session_state.sample_query:
    # st.markdown(st.session_state.sample_query)
    js_code = f"""
            <script>
            const doc = window.parent.document;
            const chatInput = doc.querySelector('.stChatInput textarea');
            chatInput.focus();

            function autoResizeTextarea() {{
                // chatInput.value = '{st.session_state.sample_query}';   
                chatInput.style.height = 'auto';
                chatInput.style.height = chatInput.scrollHeight + 'px';
                var nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, "value").set;
                nativeInputValueSetter.call(chatInput, "{st.session_state.sample_query} ");
                const event = new Event('input', {{ bubbles: true }});
                chatInput.dispatchEvent(event);

                const observer = new MutationObserver((mutations, obs) => {{
                    const clearButton = doc.querySelector('svg[title="Clear value"]');  
                    if (clearButton) {{
                        // Create and dispatch custom events
                        const mouseDown = new MouseEvent('mousedown', {{ bubbles: true }});
                        const mouseUp = new MouseEvent('mouseup', {{ bubbles: true }});
                        const click = new MouseEvent('click', {{ bubbles: true }});

                        setTimeout(() => {{
                            clearButton.dispatchEvent(mouseDown);
                            clearButton.dispatchEvent(mouseUp);
                            clearButton.dispatchEvent(click);
                        }}, 100);
                        obs.disconnect();
                    }}
                }});
                
                observer.observe(doc.body, {{
                    childList: true,
                    subtree: true
                }});

            }}
            setTimeout(autoResizeTextarea, 100);

            </script>
            """
    html(js_code)

# if map_config:
#     map_config_json = json.loads(map_config)
#     st.code(json.dumps(map_config_json, indent=4))
