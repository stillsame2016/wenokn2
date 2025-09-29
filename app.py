import time
import json
import uuid
import requests
import traceback
import streamlit as st
import pandas as pd
import geopandas as gpd
import datacommons_pandas as dc
import sparql_dataframe

from keplergl import keplergl
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from shapely import wkt
from shapely.geometry import box

from util import *
from aggregation_request import get_code_for_grouping_object, get_code_for_summarizing_object, get_aggregation_plan, get_code_for_aggregation
from refine_request import get_refined_question
from request_router import get_question_route
from request_plan import get_request_plan
from check_report_request import check_report_request, create_report_plan
from dataframe_table import render_interface_for_table
from data_commons import get_time_series_dataframe_for_dcid, get_dcid_from_county_name,  get_dcid_from_state_name, get_dcid_from_country_name, get_variables_for_dcid
from energy_atlas import *
from wenokn_use_energy_atlas import process_wenokn_use_energy_atlas

from streamlit.components.v1 import html

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup LLM
Groq_KEY = st.secrets["Groq_KEY"]
Groq_KEY_2 = st.secrets["Groq_KEY_2"]
OpenAI_KEY = st.secrets["OpenAI_KEY"]

# llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY)
# llm2 = ChatGroq(temperature=0, model_name="llama3-70b-8192", api_key=Groq_KEY_2)

llm = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=Groq_KEY)
llm2 = ChatGroq(temperature=0, model_name="llama-3.1-70b-versatile", api_key=Groq_KEY_2)

# gpt-4o
llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_tokens=5000, api_key=OpenAI_KEY)
llm2 = ChatOpenAI(model="gpt-4.1-mini", temperature=0, max_tokens=5000, api_key=OpenAI_KEY)

# Set the wide layout of the web page
st.set_page_config(layout="wide", page_title="WEN-OKN")

# Set up the title
# st.markdown("### &nbsp; WEN-OKN: Dive into Data, Never Easier")
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
    try:
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
                # If a DataFrame in the cache is not on the map, that DataFrame is considered likely to have been deleted.
                #
                # However, newly stored DataFrames in the cache do not appear on the map immediately, and such newly stored 
                # DataFrames are also mistakenly recognized as deleted.
                #
                # In order to distinguish between the above two cases, each DataFrame is timestamped with the time it was 
                # deposited, and we stipulate that a DataFrame will not be deleted for 10 seconds after it has been deposited.
                #
                if time.time() - st.session_state.datasets[i].time > 10:  
                    # st.code(f"{time.time() - st.session_state.datasets[i].time}")
                    del st.session_state.datasets[i]
                    del st.session_state.requests[i]
                    del st.session_state.sparqls[i]
                    deleted = True
            if deleted:
                # time.sleep(10)    
                st.rerun()
        return _map_config
    except Exception as e:
        logging.error(f"Error in add_map fragment: {str(e)}")
        st.error("The session expired. Please reload the web app to start.")
        return None

def ordinal(n):
    suffix = ['th', 'st', 'nd', 'rd', 'th', 'th', 'th', 'th', 'th', 'th']
    if n % 100 in [11, 12, 13]:  # Special case for 11th, 12th, 13th
        return f"{n}th"
    return f"{n}{suffix[n % 10]}"

def execute_query(user_input, chat_container):
    logger.info(f"execute_query: {user_input}")
    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/plan?query={user_input}")
    query_plan_text = None
    message = None
    if response.status_code == 200:
        query_plan = json.loads(response.text)
        query_plan = normalize_query_plan(query_plan)
        logger.info(f"query plan: {json.dumps(query_plan, indent=4)}")
        if len(query_plan) > 1:
            # show the query plan
            query_plan_text = "The following query plan has been designed to address your request:\n"
            for i, query in enumerate(query_plan, 1):
                query_plan_text += f"{i}. {query['request']}\n"
            st.markdown(query_plan_text)
            
            count_start = len(st.session_state.datasets)
            for i, query in enumerate(query_plan, 1):
                logger.info(f"Processing the {ordinal(i)} query in the query plan: **{query['request']}**")
                with chat_container:
                    with st.chat_message("assistant"):                   
                        st.markdown(f"Processing the {ordinal(i)} query in the query plan: **{query['request']}**")
                        # time.sleep(10)
                        if query["data_source"] == "WEN-OKN Database":
                            process_data_request(query["request"], chat_container)
                        elif query["data_source"] == "Data Commons":
                            code = process_data_commons_request(llm, user_input, st.session_state.datasets)
                            code = strip_code(code)
                            logger.info(f"Code:\n {code}")
                            # st.code(code)
                            # time.sleep(10)
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
                        elif query["data_source"] == "WEN-KEN database use Energy Atlas":
                            code = process_wenokn_use_energy_atlas(llm, query["request"])
                            code = strip_code(code)
                            # st.code(code)
                            # time.sleep(10)
                            globals_dict = {
                                'st': st,
                                'box': box,
                                'gpd': gpd,
                                'process_data_request': process_data_request,
                                'get_gdf_from_data_request': get_gdf_from_data_request,
                                'chat_container': chat_container,
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
                            # try:
                            #     exec(code, globals_dict)
                            # except Exception as e:
                            #     st.code(code)
                            #     error_stack = traceback.format_exc()
                            #     st.code(error_stack)
                            #     time.sleep(20)

                            if "converted_request" in globals_dict:
                                converted_request = globals_dict['converted_request']
                                if converted_request:
                                    st.markdown(f"Loaded data from ArcGIS Feature Service and converted the request to: {converted_request}")
                                    process_data_request(converted_request, chat_container)
                                    st.session_state.datasets[-1].label = query["request"]
                                    st.session_state.requests[-1] = query["request"]
                                    
                            if "gdf" in globals_dict:
                                gdf = globals_dict['gdf']
                                if gdf is not None and not gdf.empty: 
                                    gdf.label = query["request"]
                                    gdf.id = str(uuid.uuid4())[:8]
                                    gdf.time = time.time()
                                    st.session_state.requests.append(query["request"])
                                    st.session_state.sparqls.append("")
                                    st.session_state.datasets.append(gdf)

                                    message = f"""
                                                Your request has been processed. {gdf.shape[0]} 
                                                { "items are" if gdf.shape[0] > 1 else "item is"}
                                                loaded on the map.
                                                """
                                                                    
                            # st.session_state.datasets[-1].label = query["request"]
                            # st.session_state.requests[-1] = query["request"]
                        
                        elif query["data_source"] == "Energy Atlas":
                            code = process_energy_atlas_request(llm, query["request"], st.session_state.datasets)
                            code = strip_code(code)
                            logger.info(f"created code: {code}")
                            globals_dict = {
                                'st': st,
                                'gpd': gpd,
                                'pd': pd,
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
                                'load_biodiesel_plants': load_biodiesel_plants,
                                'load_watersheds': load_watersheds,
                                'load_basins': load_basins,
                                'load_census_block': load_census_block,
                                'load_nearby_census_blocks': load_nearby_census_blocks,
                                'load_census_tract': load_census_tract,
                                'downstream_tracts': downstream_tracts,
                                'load_flooded_power_stations': load_flooded_power_stations,
                                'load_flooded_buildings': load_flooded_buildings,
                                'fetch_flood_impacts': fetch_flood_impacts,
                                'load_public_water_systems': load_public_water_systems,
                                'load_PFAS_contamiation_observations': load_PFAS_contamiation_observations, 
                                'load_FRS_facilities': load_FRS_facilities
                            }
                            exec(code, globals_dict)
                            gdf = globals_dict['gdf']
                            logger.info(f"fetched geodataframe columns: {gdf.columns.to_list()}")
                            logger.info(f"fetched geodataframe: {gdf.shape}")
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
                logger.info(f"reset st.session_state.datasets[{idx}].time: {st.session_state.datasets[idx].time}")
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
                st.code(normal_print(sparql))

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
        'Show Ross County in Ohio State.', 
        'Show all counties in Kentucky State.', 
        'Find all counties the Scioto River flows through.',
        'Find all counties downstream of Ross County on the Scioto River.',  
        'Find all counties both the Ohio River and the Muskingum River flow through.',  
        'Find all counties downstream of the coal mine with the name Bowser Mine along Ohio River.',
        'Find all neighboring counties of Guernsey County.',
        'Find all adjacent states to the state of Ohio.',

        ######## River ########
        'Show the Ohio River.', 
        'Find all rivers that flow through Ross County.', 
        'What rivers flow through Dane County in Wisconsin?', 

        ######## Gages ########
        'Show all stream gauges on Muskingum River', 
        'Show all stream gages in Ross county in Ohio',
        'What stream gages are on the Yahara River in Madison, WI?',  
        'Find all stream gages on the Yahara River, which are not in Madison, WI',

        ######## Dam ########
        'Find all dams on the Ohio River.', 
        'Find all dams in Kentucky State.',
        'Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river',
        
        ######## Data Commons ########
        'Show the populations for all counties in Ohio State.', 
        'Find populations for all adjacent states to the state of Ohio.',
        'Find the median individual income for Ross County and Scioto County.', 
        'Find the number of people employed in all counties the Scioto River flows through.', 
        "Show social vulnerability index of all counties downstream of coal mine with the name 'Bowser Mine' along Ohio River",

        ######## Energy Atlas ########
        'Find all solar power plants in California.', 
        'Find all coal mines along the Ohio River.', 
        'Where are the coal-fired power plants in Kentucky?',
        'Show natural gas power plants in Florida.',
        'Load all wind power plants with total megawatt capacity greater than 100 in California.' ,

        ######## Basin ########
        'Find the basin Lower Ohio-Salt',
        'Find all basins through which the Scioto River flows.',
        'Find all rivers that flow through the Roanoke basin.',
        'Find all watersheds in the Kanawha basin.',
        # 'Find all basins with an area greater than 200,000 square kilometers',
        
        ######## Watershed ########
        'Find all watersheds feed into Muskingum River',
        'Find all watersheds in Ross County in Ohio State',
        'Find the watershed with the name Headwaters Black Fork Mohican River',
        'Find all stream gages in the watershed with the name Meigs Creek',
        'Find all stream gages in the watersheds feed into Scioto River',
        'Find all rivers that flow through the watershed with the name Headwaters Auglaize River',

        ######## NPDES ########
        'How do I determine if my facility is subject to NPDES regulations in Ohio?',

        ######## Aggregation ########
        'How many rivers flow through each county in Ohio?',
        'How many dams are in each county of Ohio?',
        'How many coal mines are in each basin in Ohio?',
        'What is the longest river in each county of Ohio state?',

        ######### Cenuse Block ########
        # 'Find the census tract at the latitude 32.7157 and longitude -117.1611',
        # 'Find the census tract for the power station dpq5d2851w52',
        # 'Find social vulnerability index of the census tract for the power station dnuzf75z9hdz',
        # 'Find the population of the census tract for the power station dnvpdhth38pu',
        # 'Find all census blocks within the 3 miles distance to the power station dpq07u6nxsp4',

        'Retrieve all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025',
        'Find the tracts of all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025', 
        'Find the populations of the tracts of all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025',
        
        'Find all power stations at risk of flooding in Ohio from 6 PM 2025-07-18 to 6 AM 2025-07-19',
        'Find social vulnerability index of the tracts of all power stations at risk of flooding in Ohio from 2 PM to 10 PM on July 12, 2025',

        'Retrieve all buildings at risk of flooding in Ohio at 2 PM on July 1, 2025',
        'Find the tracts of all buildings at risk of flooding in Ohio at 2 PM on July 1, 2025',
        'Find the populations of the tracts of all buildings at risk of flooding in Ohio at 2 PM on July 1, 2025',

        ########## PFAS Contamination Observation and Public Water System ##########
        'Find all public water systems in Maine containing PFAS contamination observations',
        'Find all PFSA contamination observations within public water systems in Maine',
        
        'Find social vulnerability index of all census tracts located downstream of the Presumpscot River from PFAS contamination observations within 100 meters of the river.',
        'Find populations of all census tracts located downstream of the Presumpscot River from PFAS contamination observations within 100 meters of the river.',
        
        'Find all public water systems in Ross county, Ohio',
        'Show all public water systems in Ohio',
        'Show all PFAS contamiation observations',
        'Find PFSA contamination observations within 20 meters to Presumpscot River',

        'Find all FRS sewage treatment facilities in Ohio',
        'Find all FRS water supply and irrigation facilities in Maine',
        'Find all FRS waste treatment and disposal facilities in Illinois',
        'Find all PFSA contamination observations within 800 meters from FRS water supply and irrigation facilities in Maine',
        'Find all FRS water supply and irrigation facilities in Maine within 800 meters from PFSA contamination observations',

        'Find FRS solid waste landfill facilities in Ohio',
        'Identify FRS solid waste landfill facilities within 1 km of Androscoggin river',
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
                                 placeholder="Select a sample query to edit and run as needed",
                                 key='selection_index')
    if selected_item:
        st.session_state.sample_query = selected_item  
        
    if user_input:
        st.session_state.sample_query = None
        with chat_container:
            st.chat_message("user").markdown(user_input)
            st.session_state.chat.append({"role": "user", "content": user_input})
            logger.info("="*50)
            
            # get initial classification
            route = get_question_route(llm, user_input)
            logger.info(f"get_question_route: {route}")

            # check whether it is a report request
            report_request = check_report_request(llm, user_input)
            logger.info(f"*report_request: {report_request}")
            
            # st.markdown(route)
            # time.sleep(30)
            if report_request['create_report']:
                report_plan = create_report_plan(llm, user_input)
                logger.info(f"report_plan: {report_plan}")
                report_query_text = ""
                for report_query in report_plan:
                    report_query_text = f"{report_query_text}\n{report_query}"                
                try:
                    query_plan_text, message = execute_query(report_query_text, chat_container)
                except Exception as error:
                    st.chat_message("assistant").markdown(f"{str(error)}")
                st.rerun()
            elif route['request_type'] == 'WEN-KEN database':
                refined_request = get_refined_question(llm, user_input)
                # st.code(refined_request)
                # time.sleep(10)
                if refined_request['is_request_data']:
                    plan = get_request_plan(llm, refined_request['request'])
                    count_start = len(st.session_state.datasets)
                    # st.code(json.dumps(plan, indent=4))
                    existed_requests = []
                    for request in plan['requests']:
                        exist_json = spatial_dataset_exists(llm, request, st.session_state.datasets)
                        # st.code(exist_json)
                        # time.sleep(10)
                        if not exist_json['existing']:
                            process_data_request(request, chat_container)
                        else:
                            existed_requests.append(request)
                            with st.chat_message("assistant"):
                                st.markdown(f"Your request has been processed. The data for the request \"{request}\" already exists.")
                                time.sleep(1)
                    count_end = len(st.session_state.datasets)   
                    for idx in range(count_start, count_end):
                        st.session_state.datasets[idx].time = time.time()

                    append_message = ""
                    if len(existed_requests) == 1:
                        append_message = f"The data for the request \"{existed_requests[0]}\" already exists."
                    elif len(existed_requests) > 1:
                        append_message = f"The data for the following requests already exists.\n"
                        for i, existed_request in enumerate(existed_requests):
                            append_message = f"{append_message}\n{i+1}. {existed_request} "
                        
                    st.session_state.chat.append({"role": "assistant",
                                                  "content": f"Your request has been processed. {append_message}"})
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
                exist_json = nonspatial_dataset_exists(llm, user_input, st.session_state.wen_datasets)
                if exist_json['existing']:
                    with st.chat_message("assistant"):
                        message = f"Your request has been processed. The data for the request \"{user_input}\" already exists."
                        st.session_state.chat.append({"role": "assistant", "content": message})
                        st.rerun()
                else:
                    code = process_data_commons_request(llm, user_input, st.session_state.datasets)
                    code = strip_code(code)
                    # st.code(f"Init Code: \n {code}")
                    # time.sleep(10)
                    with st.chat_message("assistant"):
                        with st.spinner("Loading data ..."):
                            message = "We are not able to process your request. Please refine your request and try it again."
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
                                # st.code("Init Code Failed. Generate and run a query plan")
                                # time.sleep(10)
                                try:
                                    query_plan_text, message = execute_query(user_input, chat_container)
                                except Exception as error:
                                    # message = f"""
                                    #            {code} 
                                    #            {str(e)}
                                    #            """               
                                    message = f"""We are not able to process your request. Please refine your 
                                                  request and try it again. \n\nError: {str(e)}"""
                            
                            st.markdown(message)
                            st.session_state.chat.append({"role": "assistant", "content": message})
                            st.rerun()
            elif route['request_type'] == 'US Energy Atlas' or route['request_type'] == 'Energy Atlas':
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        try:
                            exist_json = spatial_dataset_exists(llm, user_input, st.session_state.datasets)
                            # logger.info(f"exist_json:\n{exists_json}")
                            # time.sleep(10)    
                            
                            if exist_json and exist_json['existing']:
                                logger.info(f"exist_json:\n{exist_json}")
                                message = f"Your request has been processed. The data for the request \"{user_input}\" already exists."
                            else:
                                logger.info(f"processing the request for ArcGIS")
                                code = process_energy_atlas_request(llm, user_input, st.session_state.datasets)
                                code = strip_code(code)
                                logger.info(f"Code:\n{code}")
                                
                                # st.code(code)
                                # time.sleep(10)
                                exec(code)
                                # try:
                                #     exec(code)
                                # except Exception as e:
                                #     error_stack = traceback.format_exc()
                                #     st.code(error_stack)
                                #     time.sleep(20)
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
                            error_stack = traceback.format_exc()
                            logger.info(error_stack)
                            
                            # message = f"""
                            #            ERROR: {str(e)}
                            #            """  
                            # time.sleep(20)
                            message = f"""We are not able to process your request. Please refine your 
                                          request and try it again. \n\nError: {str(e)}"""
                            try:
                                query_plan_text, message = execute_query(user_input, chat_container)
                            except Exception as error:
                                message = f"{str(error)}"
                                pass
             
                    st.markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
            elif route['request_type'] == "WEN-KEN database use Energy Atlas":
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        try:
                            code = process_wenokn_use_energy_atlas(llm, user_input)
                            code = strip_code(code)
                            if "converted_request = None" in code:
                                raise ValueError("Found no converted request.")
                            # st.code(code)
                            # time.sleep(20)
                            exec(code)
                            st.markdown(f"Loaded data from ArcGIS Feature Service and converted the request to: {converted_request}")
                            process_data_request(converted_request, chat_container)
                            st.session_state.datasets[-1].label = user_input
                            st.session_state.requests[-1] = user_input
                            message = "Your request has been processed."   
                        except Exception as e:
                            try:
                                query_plan_text, message = execute_query(user_input, chat_container)
                            except Exception as error:
                                # error_stack = traceback.format_exc()
                                # message = f"""
                                #            {code} 
                                #            {error_stack}
                                #            """               
                                message = f"{str(error)}"
                    st.markdown(message)
                    st.session_state.chat.append({"role": "assistant", "content": message})
                    # st.rerun()
                    st.session_state.rerun = True
            elif route['request_type'] == 'Aggregation':
                with st.chat_message("assistant"):
                    with st.spinner("Loading data ..."):
                        try:
                            # get aggregation plan
                            aggregation_info = get_aggregation_plan(llm, user_input)
                            logger.info(json.dumps(aggregation_info, indent=4))
                            
                            # show the aggregation plan
                            aggregation_plan_text = "**The following query plan has been designed to address your aggregation request:**\n"
                            for i, query in enumerate(aggregation_info["query_plan"], 1):
                                aggregation_plan_text += f"{i}. {query['request']}\n"
                            st.markdown(aggregation_plan_text)
            
                            globals_dict = {    
                                'st': st,
                                'sparql_dataframe': sparql_dataframe,
                                'to_gdf': to_gdf,
                                'gpd': gpd,
                                'load_coal_mines': load_coal_mines,
                                'load_basins_2': load_basins_2,
                                
                                "get_variables_for_dcid": get_variables_for_dcid,
                                "get_time_series_dataframe_for_dcid": get_time_series_dataframe_for_dcid,
                                "get_dcid_from_county_name": get_dcid_from_county_name,
                                "get_dcid_from_state_name": get_dcid_from_state_name,
                                "get_dcid_from_country_name": get_dcid_from_country_name,

                                'load_coal_mines': load_coal_mines,
                                'load_coal_power_plants': load_coal_power_plants,
                                'load_wind_power_plants': load_wind_power_plants,
                            }
                            
                            # -------------------------------------------
                            # get the code for fetching group_object
                            grouping_object_request = aggregation_info["query_plan"][0]
                            logger.info(f"Process the grouping request: {json.dumps(grouping_object_request, indent=4)}")
                            st.markdown(f"**Process the grouping request**: {grouping_object_request['request']}")
                            
                            # execute the query plan for the grouping objects request
                            max_tries = 3
                            current_try = 0
                            while current_try < max_tries:
                                try:
                                    query_plan_text, message = execute_query(grouping_object_request['request'], chat_container)
                                    if query_plan_text:
                                        logger.info(f"execute_query return: {query_plan_text}")
                                        logger.info(f"execute_query return: {message}")
                                        st.session_state.rerun = False
                                        grouping_gdf = st.session_state.datasets[-1]
                                    else:
                                        # process the request
                                        code_for_grouping_object = get_code_for_grouping_object(llm, grouping_object_request)
                                        code_for_grouping_object = code_for_grouping_object.replace("load_basins(", "load_basins_2(")
                                        logger.info(code_for_grouping_object)
                                        st.markdown(f"**Executing the following code:**")
                                        st.code(code_for_grouping_object)
            
                                        # fetch grouping objects and their bounding box
                                        exec(code_for_grouping_object, globals_dict)  
                                        if 'grouping_gdf' in globals_dict.keys():
                                            grouping_gdf = globals_dict['grouping_gdf']  
                                        else:
                                            grouping_gdf = globals_dict['gdf']
                                            grouping_gdf.label = grouping_object_request
                                    break
                                except Exception as error:
                                    current_try += 1
                                    if current_try == max_tries:
                                        raise error
                                    else:
                                        st.markdown(f"Encounter an error '{str(error)}'. Try again.")
                                    
                            logger.info(f"Columns for the fetched grouping objects: {grouping_gdf.columns.to_list()}")
                            logger.info(f"The Shape for the fetched grouping objects: {grouping_gdf.shape}")
                            st.markdown(f"**The grouping obejcts are fetched:** {grouping_gdf.shape} rows")
                                                    
                            grouping_bbox = grouping_gdf.total_bounds
                            describe_bbox = lambda bbox: f"from ({bbox[0]:.4f}, {bbox[1]:.4f}) to ({bbox[2]:.4f}, {bbox[3]:.4f})"
                            st.markdown(f"**The bounding box of the grouping objects for optimizing:**")
                            st.code(f"{describe_bbox(grouping_bbox)}")

                            # -------------------------------------------
                            # get the code for fetching summarizing object
                            summarizing_object_request = aggregation_info["query_plan"][1]
                            logger.info(f"Process the summarizing request: {json.dumps(summarizing_object_request, indent=4)}")
                            st.markdown(f"**Process the summarizing request**: {summarizing_object_request['request']}")

                            request_copy = summarizing_object_request.copy()
                            if request_copy["data_source"] == "WEN-OKN database":
                                request_copy["request"] = request_copy["request"].replace("Find all", "Find 10000")

                            max_tries = 3
                            current_try = 0
                            while current_try < max_tries:
                                try:
                                    code_for_summarizing_object = get_code_for_summarizing_object(llm, request_copy, grouping_bbox)
                                    logger.info(code_for_summarizing_object)
                                    st.markdown(f"**Executing the following code:**")
                                    st.code(code_for_summarizing_object)
        
                                    # fetch summarizing objects
                                    exec(code_for_summarizing_object, globals_dict)    
                                    if 'summarizing_object_gdf' in globals_dict.keys():
                                        summarizing_object_gdf = globals_dict['summarizing_object_gdf']   
                                    else:
                                        summarizing_object_gdf = globals_dict['gdf']
                                        summarizing_object_gdf.label = summarizing_object_request
                                    break
                                except Exception as error:
                                    current_try += 1
                                    if current_try == max_tries:
                                        raise error
                                    else:
                                        st.markdown(f"Encounter an error '{str(error)}'. Try again.")

                            logger.info(f"Columns for the summarizing grouping objects: {summarizing_object_gdf.columns.to_list()}")
                            logger.info(f"The Shape for the summarizing grouping objects: {summarizing_object_gdf.shape}")
                            st.markdown(f"**The summarizing obejcts are fetched:** {summarizing_object_gdf.shape} rows")

                            # Fix CRS if it doesn't match the geometry
                            if detect_4326_in_3857(summarizing_object_gdf) and summarizing_object_gdf.crs != "EPSG:4326":
                                summarizing_object_gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

                            # -------------------------------------------
                            # resolve the aggregation request
                            logger.info(f"Process aggregation: {user_input}")
                            st.markdown(f"**Process aggregation:** {user_input}")
                            
                            code_for_aggregation = strip_code(get_code_for_aggregation(llm, grouping_gdf, summarizing_object_gdf, user_input))
                            logger.info(code_for_aggregation)
                            st.markdown(f"**Executing the following code:**")
                            st.code(code_for_aggregation)
                            
                            globals_dict['grouping_gdf'] = grouping_gdf
                            globals_dict['summarizing_object_gdf'] = summarizing_object_gdf
                            exec(code_for_aggregation, globals_dict)
                            df = globals_dict['df']
                            df.id = user_input
                            df.title = user_input
                            st.session_state.wen_datasets.append(df)
                            st.session_state.wen_tables.append(df.copy())
                            st.session_state.table_chat_histories.append([])
                            st.session_state.chart_types.append("bar_chart")

                            # -------------------------------------------
                            # setup data for the map display 
                            
                            grouping_gdf.title = grouping_object_request['request']
                            grouping_gdf.label = grouping_object_request['request']
                            grouping_gdf.id = str(uuid.uuid4())[:8]
                            grouping_gdf.time = time.time()
                            st.session_state.requests.append(grouping_object_request['request'])
                            st.session_state.sparqls.append("")
                            st.session_state.datasets.append(grouping_gdf)

                            logger.info(f"Summarizing CRS: {summarizing_object_gdf.crs}")
                            logger.info(f"Grouping CRS: {grouping_gdf.crs}")

                            logger.info(f"Summarizing shape: {summarizing_object_gdf.shape}")
                            logger.info(f"Grouping shape: {grouping_gdf.shape}")
                            
                            if detect_4326_in_3857(summarizing_object_gdf) and summarizing_object_gdf.crs != "EPSG:4326":
                                # First, set the CRS correctly (if the actual geometry is in EPSG:4326)
                                summarizing_object_gdf.set_crs("EPSG:4326", allow_override=True, inplace=True)

                            # Perform a spatial join to keep only rows that intersect grouping_gdf
                            gdf_intersect = gpd.sjoin(summarizing_object_gdf, grouping_gdf, how="inner", predicate="intersects")
                            logger.info(f"After 'intersects' filter: {gdf_intersect.shape}")

                            logger.info(f"Columns in gdf_intersect before subsetting: {gdf_intersect.columns}")
                            logger.info(f"Columns in summarizing_object_gdf before subsetting: {summarizing_object_gdf.columns}")

                            # Keep only the original columns from summarizing_object_gdf
                            gdf_intersect = gdf_intersect[[col for col in gdf_intersect.columns if not col.endswith('_right')]]
                            gdf_intersect = gdf_intersect.drop(columns=['OBJECTID_left'], errors='ignore')

                            logger.info(f"Columns in gdf_intersect after subsetting: {gdf_intersect.columns}")
                            logger.info(f"Shape after subsetting: {gdf_intersect.shape}")
                            logger.info(f"Columns in df after subsetting: {df.columns}")

                            # Exclude the geometry column from gdf columns
                            gdf_intersect_columns = set(gdf_intersect.columns) - {'geometry'}
                            common_columns = gdf_intersect_columns.intersection(set(df.columns))
                            if len(list(common_columns)) == 1:
                                columnA = list(common_columns)[0]
                                logger.info(f"Unique common column: {columnA}")
                                gdf_intersect = gdf_intersect[gdf_intersect[columnA].isin(df[columnA])]

                            # gdf_intersect = summarizing_object_gdf
                            gdf_intersect.title = summarizing_object_request['request']
                            gdf_intersect.label = summarizing_object_request['request']
                            gdf_intersect.id = str(uuid.uuid4())[:8]
                            gdf_intersect.time = time.time()
    
                            st.session_state.requests.append(user_input)
                            st.session_state.sparqls.append("")
                            st.session_state.datasets.append(gdf_intersect)
                            
                            message = f"""
                                    Your request has been processed. {df.shape[0]} { "rows are" if df.shape[0] > 1 else "row is"}
                                    found and displayed.
                                    """
                            st.markdown(message)
                            st.session_state.chat.append({"role": "assistant", "content": message})
                            st.rerun()
                            
                        except Exception as e:
                            st.code(f"Error: {str(e)}")
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

                const clearButton = doc.querySelector('svg[title="Clear value"]');  
                console.log("clearButton: " +clearButton);

                if (clearButton) {{
                    // Create and dispatch custom events
                    const mouseDown = new MouseEvent('mousedown', {{ bubbles: true }});
                    const mouseUp = new MouseEvent('mouseup', {{ bubbles: true }});
                    const click = new MouseEvent('click', {{ bubbles: true }});

                    clearButton.dispatchEvent(mouseDown);
                    clearButton.dispatchEvent(mouseUp);
                    clearButton.dispatchEvent(click);
                }}
            }}

            setTimeout(autoResizeTextarea, 0);

            </script>
            """
    html(js_code)

# if map_config:
#     map_config_json = json.loads(map_config)
#     st.code(json.dumps(map_config_json, indent=4))
