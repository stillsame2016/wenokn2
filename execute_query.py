
import json
import requests
import time
import streamlit as st

from util import process_data_request, process_regulation_request, process_data_commons_request
from wenokn_use_energy_atlas import process_wenokn_use_energy_atlas
from energy_atlas import *


def execute_query(user_input, chat_container, llm):
    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/plan?query={user_input}")
    if response.status_code == 200:
        query_plan = json.loads(response.text)
        st.code(json.dumps(query_plan, indent=4))
        if len(query_plan) > 1:
            count_start = len(st.session_state.datasets)
            for query in query_plan:
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(f"Executing the query plan and process the query: **{query['request']}**")
                        if query["data_source"] == "WEN-OKN Database":
                            process_data_request(query["request"], chat_container)
                        elif query["data_source"] == "Energy Atlas":
                            code = process_energy_atlas_request(llm, query["request"], st.session_state.datasets)
                            if code.startswith("```python"):
                                start_index = code.find("```python") + len("```python")
                                end_index = code.find("```", start_index)
                                code = code[start_index:end_index].strip()
                            elif code.startswith("```"):
                                start_index = code.find("```") + len("```")
                                end_index = code.find("```", start_index)
                                code = code[start_index:end_index].strip()
                            st.code(code)
                            st.code(f"check: {len(st.session_state.datasets)}")
                            exec(code, { "st": st, "load_coal_mines": load_coal_mines})
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
                                message = f"""
                                            Your request has been processed. Nothing was found.
                                            Please refine your request and try again if you think
                                            this is a mistake.
                                            """
            count_end = len(st.session_state.datasets)   
            for idx in range(count_start, count_end):
                st.session_state.datasets[idx].time = time.time()
            st.session_state.rerun = True
