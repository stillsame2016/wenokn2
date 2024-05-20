import re
import uuid
import json
import requests
import sparql_dataframe
import geopandas as gpd
from langchain_core.prompts import PromptTemplate
from shapely import wkt
import streamlit as st
from langchain_core.output_parsers import StrOutputParser


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
def process_data_request(message, chat_container):
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(f"""We're currently processing your request:
                                **{message}{'' if message.endswith('.') else '.'}**
                          Depending on the complexity of the query and the volume of data, 
                          this may take a moment. We appreciate your patience."""):

                # generate a sparql query. try up to 5 times because of the LLM limit
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
                        st.session_state.chat.append({"role": "assistant",
                                                      "content": "Your request has been processed."})

                        st.rerun()
                    except Exception as e:
                        st.markdown(f"Encounter an error: {str(e)}. Try again...")
                        # traceback.print_exc()
                        tried += 1
                if tried == max_tries:
                    st.markdown(
                        "We are not able to process your request at this moment. You can try it again now or "
                        "later.")


def process_regulation_request(llm, user_input, chat_container):
    VDB_URL = "https://sparcal.sdsc.edu/api/v1/Utility/regulations"
    KPDES_URL = "https://sparcal.sdsc.edu/api/v1/Utility/kpdes"
    template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are the expert of National Pollution 
            Discharge Elimination System (NPDES) and Kentucky Pollutant Discharge Elimination System (KPDES). 

            The National Pollutant Discharge Elimination System (NPDES) is a regulatory program implemented by the United 
            States Environmental Protection Agency (EPA) to control water pollution. It was established under the Clean 
            Water Act (CWA) to address the discharge of pollutants into the waters of the United States.

            The NPDES program requires permits for any point source that discharges pollutants into navigable waters, 
            which include rivers, lakes, streams, coastal areas, and other bodies of water. Point sources are discrete 
            conveyances such as pipes, ditches, or channels.

            Under the NPDES program, permits are issued to regulate the quantity, quality, and timing of the pollutants 
            discharged into water bodies. These permits include limits on the types and amounts of pollutants that can 
            be discharged, monitoring and reporting requirements, and other conditions to ensure compliance with water 
            quality standards and protect the environment and public health.

            The goal of the NPDES program is to eliminate or minimize the discharge of pollutants into water bodies, 
            thereby improving and maintaining water quality, protecting aquatic ecosystems, and safeguarding human health. 
            It plays a critical role in preventing water pollution and maintaining the integrity of the nation's water 
            resources.

            Based on the provided context, use easy understanding language to answer the question clear and precise with 
            references and explanations. If the local regulations (for example, KPDES for Kentucky Pollutant Discharge 
            Elimination System) can be applied, please include the details of both NPDES rules and KPDES rules, and make 
            clear indications of the sources of the rules.

            If no information is provided in the context, return the result as "Sorry I dont know the answer", don't provide 
            the wrong answer or a contradictory answer.

            Context:{context}

            Question:{question}?

            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question", "context"],
    )
    rag_chain = template | llm | StrOutputParser()

    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("We are in the process of retrieving the relevant provisions "
                            "to give you the best possible answer."):
                if "kentucky" in user_input.lower() or "KPDES" in user_input:
                    response = requests.get(f"{VDB_URL}?search_terms={user_input}")
                    datasets = json.loads(response.text)
                    datasets = datasets[0:4]
                    context = "NPDES regulations: "
                    context += "\n".join([dataset["description"] for dataset in datasets])

                    response = requests.get(f"{KPDES_URL}?search_terms={user_input}")
                    datasets = json.loads(response.text)
                    datasets = datasets[0:4]
                    context += "\nKPDES (Kentucky Pollutant Discharge Elimination System) regulations: "
                    context += "\n".join([dataset["description"] for dataset in datasets])
                else:
                    response = requests.get(f"{VDB_URL}?search_terms={user_input}")
                    datasets = json.loads(response.text)
                    datasets = datasets[0:5]
                    context = "\n".join([dataset["description"] for dataset in datasets])

                result = rag_chain.invoke({"question": user_input, "context": context})
                return result


def process_off_topic_request(llm, user_input, chat_container):
    template = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert of the WEN-OKN 
            knowledge database and National Pollution Discharge Elimination System (NPDES) and Kentucky 
            Pollutant Discharge Elimination System (KPDES). 

            The WEN-KEN database contains the following entities: 
              1. Locations of buildings, power stations, and underground storage tanks in Ohio.
              2. USA Counties: names and geometry boundaries.
              3. USA States: names and geometry boundaries.
              4. Earthquakes: Data pertaining to seismic events.
              5. Rivers: Comprehensive geometries about rivers in USA.
              6. Dams: Information regarding dams' locations in USA.
              7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
              8. Hospitals: Details about hospital locations and information in USA.

            The National Pollutant Discharge Elimination System (NPDES) is a regulatory program implemented by the United 
            States Environmental Protection Agency (EPA) to control water pollution. It was established under the Clean 
            Water Act (CWA) to address the discharge of pollutants into the waters of the United States.

            The NPDES program requires permits for any point source that discharges pollutants into navigable waters, 
            which include rivers, lakes, streams, coastal areas, and other bodies of water. Point sources are discrete 
            conveyances such as pipes, ditches, or channels.

            Under the NPDES program, permits are issued to regulate the quantity, quality, and timing of the pollutants 
            discharged into water bodies. These permits include limits on the types and amounts of pollutants that can 
            be discharged, monitoring and reporting requirements, and other conditions to ensure compliance with water 
            quality standards and protect the environment and public health.

            The goal of the NPDES program is to eliminate or minimize the discharge of pollutants into water bodies, 
            thereby improving and maintaining water quality, protecting aquatic ecosystems, and safeguarding human health. 
            It plays a critical role in preventing water pollution and maintaining the integrity of the nation's water 
            resources.

            Based on the provided context, use easy understanding language to answer the question politely.
            
            Question:{question}?

            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    rag_chain = template | llm | StrOutputParser()

    with chat_container:
        with st.chat_message("assistant"):
            result = rag_chain.invoke({"question": user_input})
            return result
