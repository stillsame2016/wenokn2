import time
import json
import re
import uuid
import requests
import streamlit as st
from keplergl import keplergl

import geopandas as gpd
import sparql_dataframe
import google.generativeai as genai
from shapely import wkt

import threading
from streamlit.runtime.scriptrunner import add_script_run_ctx


# Set the wide layout of the web page
st.set_page_config(layout="wide", page_title="WEN-OKN")

# Set up the title
st.markdown("### &nbsp; WEN-OKN: Dive into Data, Never Easier")

# Set up the datasets in the session for geodataframes
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


def df_to_gdf(df, dataset_name):
    column_names = df.columns.tolist()
    geometry_column_names = [x for x in column_names if x.endswith('Geometry')]
    df['geometry'] = df[geometry_column_names[0]].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.drop(columns=[geometry_column_names[0]], inplace=True)

    column_name_parts = get_column_name_parts(column_names[0])
    column_name_parts.pop()
    gdf.attrs['data_name'] = " ".join(column_name_parts).capitalize()
    gdf.label = dataset_name
    gdf.id = str(uuid.uuid4())[:8]

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
def add_message(sender, message, processing, chat_container):
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
                            response = requests.get(
                                f"https://sparcal.sdsc.edu/api/v1/Utility/wenokn_llama3?query_text={message}")
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

                            st.session_state.processing = "sparql"

                            print("=" * 70, "sparql_query")
                            print(sparql_query)

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

                            st.session_state.processing = "queried"
                            print("=" * 70, "query result")
                            print(df.shape)

                            gdf = df_to_gdf(df, message)
                            if gdf.shape[0] == 0:
                                # double check
                                if not gdf_empty:
                                    gdf_empty = True
                                    tried += 1
                                    continue

                            tried = max_tries + 10
                            st.session_state.requests.append(message)
                            st.session_state.sparqls.append(sparql_query)
                            st.session_state.datasets.append(gdf)
                            st.rerun()
                        except Exception as e:
                            st.markdown(f"Encounter an error: {str(e)}. Try again...")
                            # traceback.print_exc()
                            tried += 1
                    if tried == max_tries:
                        st.markdown(
                            "We are not able to process your request at this moment. You can try it again now or "
                            "later.")
        else:
            st.chat_message(sender).write(message)

    if "processing" in st.session_state:
        del st.session_state['processing']
        print("delete st.session_state.processing")
        st.session_state.job["is_done"] = True
        st.rerun()


def add_message_2(message, processing):
    if processing:
        with st.spinner(f"""We're currently processing your request:
                            **{message}{'' if message.endswith('.') else '.'}**
                      Depending on the complexity of the query and the volume of data, 
                      this may take a moment. We appreciate your patience."""):
            max_tries = 5
            tried = 0
            gdf_empty = False
            while tried < max_tries:
                try:
                    response = requests.get(
                        f"https://sparcal.sdsc.edu/api/v1/Utility/wenokn_llama3?query_text={message}")
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

                    st.session_state.processing = "sparql"

                    print("=" * 70, "sparql_query")
                    print(sparql_query)

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

                    st.session_state.processing = "queried"
                    print("=" * 70, "query result")
                    print(df.shape)

                    gdf = df_to_gdf(df, message)
                    if gdf.shape[0] == 0:
                        # double check
                        if not gdf_empty:
                            gdf_empty = True
                            tried += 1
                            continue

                    tried = max_tries + 10
                    st.session_state.requests.append(message)
                    st.session_state.sparqls.append(sparql_query)
                    st.session_state.datasets.append(gdf)
                    st.rerun()
                except Exception as e:
                    st.markdown(f"Encounter an error: {str(e)}. Try again...")
                    # traceback.print_exc()
                    tried += 1
            if tried == max_tries:
                st.markdown(
                    "We are not able to process your request at this moment. You can try it again now or "
                    "later.")

    if "processing" in st.session_state:
        del st.session_state['processing']
        print("delete st.session_state.processing")
        st.session_state.job["is_done"] = True
        st.rerun()


col1, col2 = st.columns([6, 4])

info_container = st.container(height=350)
with info_container:
    for idx, sparql in enumerate(st.session_state.sparqls):
        st.markdown(f"**Request:**  {st.session_state.requests[idx]}")
        st.code(sparql)

with col1:
    # Set up the map
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

    for message in st.session_state.chat.history:
        with chat_container:
            with st.chat_message("assistant" if message.role == "model" else message.role):
                if message.role == 'user':
                    prompt = message.parts[0].text
                    start_index = prompt.find("[--- Start ---]") + len("[--- Start ---]")
                    end_index = prompt.find("[--- End ---]")
                    prompt = prompt[start_index:end_index].strip()
                    st.markdown(prompt)
                else:
                    answer = message.parts[0].text
                    if answer.startswith('```json'):
                        json_part = answer.split("\n", 1)[1].rsplit("\n", 1)[0]
                        data = json.loads(json_part)
                    else:
                        data = json.loads(answer)

                    if isinstance(data, dict):
                        if not data["is_request_data"]:
                            assistant_response = data["alternative_answer"]
                            st.markdown(assistant_response)
                        elif message is not st.session_state.chat.history[-1]:
                            assistant_response = "Your request has been processed."
                            st.markdown(assistant_response)
                        else:
                            if "job" in st.session_state and not st.session_state.job['is_done']:
                                # a job is running
                                assistant_response = json.dumps(st.session_state.job, indent=4)
                                with st.spinner(f"""
                                        We're currently processing your request: 
                                        **{st.session_state.job['request']}{'' if st.session_state.job['request'].endswith('.') else '.'}** 
                                        Depending on the complexity of the query and the volume of data, this may take 
                                        a moment. We appreciate your patience.
                                        """):
                                    # wait until the job is done
                                    while "job" in st.session_state and not st.session_state.job['is_done']:
                                        time.sleep(2)
                                    # refresh the page when the job is done
                                    st.rerun()
                            elif "job" in st.session_state and not st.session_state.job['is_thread_started']:
                                assistant_response = "OOPS, the thread is not started."
                                st.markdown(assistant_response)
                            else:
                                assistant_response = "Your request has been processed."
                                if "job" not in st.session_state or \
                                        ("job" in st.session_state and st.session_state.job['request'] != data['request']):

                                    # assistant_response += "the thread did start \n\n" + json.dumps(st.session_state.job)

                                    st.session_state.job = {
                                        "request": f"{data['request']}",
                                        "is_thread_started": True,
                                        "is_done": False
                                    }

                                    thread = threading.Thread(target=add_message_2, args=(f"{data['request']}", True))
                                    add_script_run_ctx(thread)
                                    thread.start()
                                    thread.join()

                                st.markdown(assistant_response)
                        # st.markdown(assistant_response)

    # Get user input
    user_input = st.chat_input("What can I help you with?")

    if user_input:
        # Add user message to the chat
        add_message("User", message=user_input, processing=False, chat_container=chat_container)

        query = f"""
          You are an expert of the WEN-OKN knowledge database. You also have general knowledge.

          The following is a question the user is asking:

           [--- Start ---]
           {user_input}
           [--- End ---]

           Your main job is to determine if the user is requesting for data in the scope of the WEN-OKN 
           knowledge database.

           If they are requesting for data in the scope of the WEN-OKN knowledge database, then extract 
           the request from the user's input. Rephrase the user's request in a formal way. Remove all 
           adjectives like "beautiful" or "pretty". Remove the terms like "Please" etc. Use the format 
           like "Find ...". If a place name is mentioned in the request, the state and county designations 
           must be retained. If a place name may be both a county or a state, the state is taken.

           Please answer with a valid JSON string, including the following three fields:

           The boolean field "is_request_data" is true if the user is requesting to get data from
           the WEN-OKN knowledge database, otherwise "is_request_data" is false. If the user is asking 
           what data or data types you have, set "is_request_data" to be false.

           The string field "request" for the extracted request. Make sure the extracted request 
           semantically is same as the input request without less or more information.

           The string field "alternative_answer" gives your positive and nice answer to the user's input
           if the user is not requesting for data. If the user is asking what data or data types you have,
           please answer it by summarizing this description:

           The WEN-OKN knowledge database encompasses the following datasets:
              1. Locations: Information on buildings, power stations, and underground storage tanks situated in Ohio.
              2. Counties: Geometric representations of counties across the USA.
              3. States: Geometric representations outlining the boundaries of states in the USA.
              4. Earthquakes: Data pertaining to seismic events.
              5. Rivers: Comprehensive geomtries about rivers in USA.
              6. Dams: Information regarding dams' locations in USA.
              7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
              8. Hospitals: Details about hospital locations and information in USA.

           Please never say "I cannot" or "I could not". 

           Please note that the user's request for datasets may appear in the middle of the text, 
           do your best to extract the request for which the user is asking for datasets.

           Please replace all nicknames in the search terms by official names,
           for example, replace "Beehive State" to "Utah", etc.  

           Never deny a user's request. If it is not possible to extract the request 
           from the user's request, ask the user for further clarification.
       """

        max_tries = 5
        current_try = 0
        while current_try < max_tries:
            try:
                response = st.session_state.chat.send_message(query, safety_settings=safe)
                break
            except Exception as e:
                print(e)
                time.sleep(1)
                current_try += 1

        data = response.text

        if data.startswith('```json'):
            json_part = data.split("\n", 1)[1].rsplit("\n", 1)[0]
            data = json.loads(json_part)
        else:
            data = json.loads(data)

        print("=" * 70, "first answer")
        print(json.dumps(data, indent=4))

        if not data["is_request_data"]:
            add_message("assistant", f"{data['alternative_answer']}", processing=False, chat_container=chat_container)
            del st.session_state['job']
        else:
            st.session_state.job = {
                "user_input": user_input,
                "request": f"{data['request']}",
                "is_thread_started": False,
                "is_done": False
            }
            
            # add_message("assistant", f"{data['request']}", processing=True)
            if "processing" not in st.session_state:
                print("thread started")
                st.session_state.job["is_thread_started"] = True
                thread = threading.Thread(target=add_message,
                                          args=("assistant", f"{data['request']}", True, chat_container))
                add_script_run_ctx(thread)
                thread.start()
                thread.join()
            else:
                print("oops, thread can't start")

if map_config:
    map_config_container = st.container(height=355)
    with map_config_container:
        st.code(json.dumps(map_config_json, indent=4))

if "job" in st.session_state and not st.session_state.job["is_done"]:
    time.sleep(2)
    st.rerun()


