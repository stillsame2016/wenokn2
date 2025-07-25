import re
import uuid
import json
import time
import requests
import sparql_dataframe
import pandas as pd
import geopandas as gpd
from langchain_core.prompts import PromptTemplate
from shapely import wkt
import streamlit as st
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser


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
    gdf.time = time.time()

    for column_name in column_names:
        tmp_column_name_parts = get_column_name_parts(column_name)
        tmp_name = tmp_column_name_parts.pop()
        tmp_data_name = " ".join(column_name_parts).capitalize()
        if gdf.attrs['data_name'] == tmp_data_name:
            gdf.rename(columns={column_name: tmp_name}, inplace=True)
            # if tmp_data_name == gdf.attrs['data_name']:
            #     gdf.rename(columns={column_name: name}, inplace=True)
    return gdf

def to_gdf(df, dataset_name):
    column_names = df.columns.tolist()
    geometry_column_names = [x for x in column_names if x.endswith('Geometry')]
    df['geometry'] = df[geometry_column_names[0]].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(df, geometry='geometry')
    gdf.drop(columns=[geometry_column_names[0]], inplace=True)
    gdf.label = dataset_name
    return gdf

# Function to add a new message to the chat
def process_data_request(message, chat_container):
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(f"""We're currently processing your request:
                                **{message}{'' if message.endswith('.') else '.'}**
                          Depending on the complexity of the query and the volume of data, 
                          this may take a moment. We appreciate your patience."""):

                # generate a sparql query. try up to 8 times because of the LLM limit
                max_tries = 8
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
                        elif data.startswith("```sql"):
                            start_index = data.find("```sql") + len("```sql")
                            end_index = data.find("```", start_index)
                            sparql_query = data[start_index:end_index].strip()
                        elif data.startswith("sql"):
                            start_index = data.find("sql") + len("sql")
                            sparql_query = data[start_index:].strip()
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
                        st.code(normal_print(sparql_query))

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
                        if gdf.shape[0] > 0:
                            st.session_state.requests.append(message)
                            st.session_state.sparqls.append(sparql_query)
                            st.session_state.datasets.append(gdf)
                        else:
                            error_info = f"""No data has been loaded for your request **{message}**.
                                             Please refine your request and try it again."""
                            st.markdown(error_info)
                            st.session_state.chat.append({"role": "assistant", "content": error_info})
                            st.rerun()
                        # st.session_state.chat.append({"role": "assistant",
                        #                               "content": "Your request has been processed."})
                        # st.rerun()
                    except Exception as e:
                        st.markdown(f"Encounter an error: {str(e)}.")
                        st.markdown("Try again...")
                        # traceback.print_exc()
                        tried += 1
                if tried == max_tries:
                    error_info = f"""We are not able to process your request **{message}** 
                                     at this moment. Please refine your request and try it again."""
                    st.markdown(error_info)
                    st.session_state.chat.append({"role": "assistant", "content": error_info})
                    st.rerun()


def process_data_commons_request(llm, user_input, spatial_datasets):    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        
        In Data Commons, dcid is used as index to access data. A dcid has the following format, 
        for example, "geoid/39" is the dcid for the Ohio State and "geoid/06" is the dcid for the
        California State. 
        
        We have the following functions to get dcid from a state/county name:
            get_dcid_from_state_name(state_name)
            get_dcid_from_county_name(county_name) 
            get_dcid_from_country_name(country_name)
        To call get_dcid_from_county_name, the county name must be in the format "San Diego County". 
        Don't miss "County" in the name. 
        
        Data Commons has the following statistical variables available for a particular place:
            {dc_variables}

        The following are all dataframes in the data repository:
            {variables}
              
        The following code can fetch some variables data for some dcid from Data Commons:
                
            import datacommons_pandas as dc
            
            def get_time_series_dataframe_for_dcid(dcid_list, variable_name):
                _df = dc.build_time_series_dataframe(dcid_list, variable_name)    
                _df.insert(0, 'Name', _df.index.map(dc.get_property_values(_df.index, 'name')))
                _df['Name'] = _df['Name'].str[0]    
                
                # columns = _df.columns.to_list().remove('Name')
                columns = _df.columns.to_list()
                columns.remove('Name')
                _df = _df.melt(
                    ['Name'],
                    columns,
                    'Date',
                     variable_name,
                )
                _df = _df.dropna()     
                _df = _df.drop_duplicates(keep='first')
                _df.variable_name = variable_name
                return _df
                
        [Example 1] 
        Find the populations for all counties in Ohio, we can run the following code:
        
            # Get dcid for all counties in Ohio
            ohio_county_dcid = dc.get_places_in(["geoId/39"], 'County')["geoId/39"]
            
            # Get Count_Person (i.e., population) for all counties in Ohio
            df = get_time_series_dataframe_for_dcid(ohio_county_dcid, "Count_Person")
            df.title = "The Populations for All Counties in Ohio"
                    
        [Example 2]
        Find the populations for the Ross county and Pike county in Ohio, we can run the 
        following code:
        
            ross_pike_dcid = ['geoId/39131', 'geoId/39141']
            df = get_time_series_dataframe_for_dcid(ross_pike_dcid, "Count_Person")
            df.title = "The Populations for the Ross county and Pike county in Ohio"
                     
        [Example 3]
        Find the populations of Ross county and Scioto county
        
            ross_scioto_dcid = [ get_dcid_from_county_name('Ross County'), get_dcid_from_county_name('Scioto County') ]
            df = get_time_series_dataframe_for_dcid(ross_scioto_dcid, "Count_Person")
            df.title = "The Populations for the Ross county and Scioto county in Ohio"
                 
        [Example 4]   
        Find the populations of all counties where Scioto River flows through.

        It is your job to check each dataframe in the data repository listed above to determine an 
        index such that st.session_state.datasets[index] for "Find all counties where Scioto River flows through". 
        Never assume an index.

        If you find such index, return the following code A:
            # Note that following index must be replaced by an integer you find. 
            gdf = st.session_state.datasets[ insert index you find here ]
            scioto_river_dcid = [ get_dcid_from_county_name(county_name) for county_name in gdf['name']]
            df = get_time_series_dataframe_for_dcid(scioto_river_dcid, "Count_Person")  
            df.title = "The Populations for All Counties where Scioto River Flows Through"
       
        If you could not find such index, return the following code B:
            raise ValueError('Please load all counties where Scioto River flows through first')

        Please note that only return code A or code B. Never combine the code A and code B together.

        [Example 5]
        Find social vulnerability index of all counties downstream of coal mine with the name 'Century Mine' 
        along Ohio River.

        Check each dataframe in the data repository listed above to find an index such that st.session_state.datasets[index] 
        contains 'Find all counties downstream of the coal mine with the name "Century Mine" along Ohio River'.  
        Never assume an index.

        If index is found, return the following code A:
            # Note that following index must be replaced by an integer you find 
            # Please look at each dataframe in the data repository to determine which index should be used. 
            # You can't assume an index
            gdf = st.session_state.datasets[index]
            counties_dcid = [ get_dcid_from_county_name(county_name) for county_name in gdf['Name']]
            df = get_time_series_dataframe_for_dcid(counties_dcid, "FemaSocialVulnerability_NaturalHazardImpact")  
            df.title = "The Social Vulnerability for All Counties Downstream of the Coal Mine with the Name \"Century Mine\" along Ohio River"

        Otherwise return the code B:
            raise ValueError("Please load all counties downstream of the coal mine with the name 'Century Mine' along Ohio River first")

        Please note that only return code A or code B. Never combine the code A and code B together.

        [ Example 6 ]
        Find social vulnerability index of the census tract for the power station dpq5d2851w52.

        Check each dataframe in the data repository listed above to find an index such that st.session_state.datasets[index] 
        contains 'Find the census tract for the power station dpq5d2851w52'.  Never assume an index.

        If index is found, return the following code A:
            gdf = st.session_state.datasets[index]
            trait_dcid = [ "geoId/"+geoid for geoid in gdf['GEOID']]
            df = get_time_series_dataframe_for_dcid(counties_dcid, "FemaSocialVulnerability_NaturalHazardImpact")  
            df.title = "The Social Vulnerability for the census tract for the power station dpq5d2851w52"

        Otherwise return the code B:
            raise ValueError("Please load the census tract for the power station dpq5d2851w52 first")

        Please note that only return code A or code B. Never combine the code A and code B together.
                 
        [ Question ]
        The following is the question from the user:
        {question}

        Please use pd.merge(df1, df2, on=df1.columns.to_list[:-1]) to merge two dataframes if needed. 

        Please return the complete Python code only to implement the user's request without preamble or 
        explanation. Don't include any print statement. Don't add ``` around the code. Make a title and
        save the title in df.title. 
    
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "variables", "dc_variables"],
    )
    df_code_chain = prompt | llm | StrOutputParser()

    dc_variables = ""
    response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/data_commons?search_terms={user_input}")
    items = json.loads(response.text)
    for item in items:
        dc_variables = f"""{dc_variables}
                            variable: {item['variable']}
                            description: {item['name']}
                            
                        """
    
    variables = ""
    if spatial_datasets:
        for index, dataset in enumerate(spatial_datasets):
            variables += f"""
    st.session_state.datasets[{index}] is a geodataframe for "{st.session_state.datasets[index].label}"

    The following is the columns of st.session_state.datasets[{index}]:
        { st.session_state.datasets[index].dtypes }

    The following is the first 5 rows of the data:
        { st.session_state.datasets[index].head(5).drop(columns='geometry').to_csv(index=False) }            
"""
    # st.code(variables)
    # time.sleep(10)
    
    return df_code_chain.invoke({"question": user_input, "variables": variables, "dc_variables": dc_variables})
    

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
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert of following systems:
               1. The WEN-OKN knowledge database 
               2. National Pollution Discharge Elimination System (NPDES) and Kentucky Pollutant Discharge Elimination System (KPDES) 
               3. Data Commons
               4. US Energy Atlas

            The WEN-KEN database contains the following entities: 
              1. Locations of buildings, power stations, and underground storage tanks in Ohio.
              2. USA Counties: names and geometry boundaries.
              3. USA States: names and geometry boundaries.
              4. Earthquakes: Data pertaining to seismic events.
              5. Rivers: Comprehensive geometries about rivers in USA.
              6. Dams: Information regarding dams' locations in USA.
              7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
              8. Hospitals: Details about hospital locations and information in USA.
              9. Stream Gages: Information of gages' locations and names in USA.

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

            Data Commons has the following data for counties or states or countries. 
                Area_FloodEvent
                Count_Person (for population)
                Count_FireEvent
                Count_FlashFloodEvent
                Count_FloodEvent
                Count_HailEvent
                Count_HeatTemperatureEvent
                Count_HeatWaveEvent
                Count_HeavyRainEvent
                CountOfClaims_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
                Max_Rainfall
                Max_Snowfall
                SettlementAmount_NaturalHazardInsurance_BuildingContents_FloodEvent
                SettlementAmount_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
                SettlementAmount_NaturalHazardInsurance_BuildingStructure_FloodEvent

            The US Energy Atlas has the following data:
                Battery Storage Plant
                Coal Mine
                Coal Power Plant
                Geothermal Power Plant
                Wind Power Plant
                Renewable Diesel Fuel and Other Biofuel Plant
                Wind Power Plant
                Hydro Pumped Storage Power Plant
                Natural Gas Power Plant
                Nuclear Power Plant
                Petroleum Power Plant
                Solar Power Plant
                Biodiesel Plant

            Based on the provided context, use easy understanding language to answer the question.
            Please politely reject any requests for searching any websites. 
            
            Question:{question}?

            Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
        input_variables=["question"],
    )
    rag_chain = template | llm | StrOutputParser()

    with chat_container:
        with st.chat_message("assistant"):
            result = rag_chain.invoke({"question": user_input})
            return result


def process_table_request(llm, llm2, user_input, index):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
            You are an expert of {title} which is loaded in a DataFrame st.session_state.wen_datasets[{index}] 
            with the following columns and types: 
                {columns}
            The following is the first 5 rows of the data:
                {sample}
            
            The data is displayed to the user.

            Users can make queries in natural language to request data from this DataFrame, or they may ask other types of 
            questions.

            Please categorize the following user question as either a "Request data" or "Other" in a JSON field "category". 

            For "Request data", return a python statement in the following format:
                 other code with with st.session_state.wen_datasets[{index}] only
                 st.session_state.wen_tables[{index}] = <your expression with st.session_state.wen_datasets[{index}] only>  
            in the JSON field "answer". Also return a short title by summarizing "{title}" and "{question}" in the JSON field 
            "title".
            
            Note that you can't use df.resample('Y', on='Time') when the type of df['Time'] is string.

            Dont use triple quotes in the JSON string which make the result an invalid JSON string. 

            For "Other", return a reasonable answer in the JSON field "answer". 

            Return JSON only without any explanations. 

            [ Example ]
            To find the date with greatest increment for each place,  we can use the following code:
            if st.session_state.wen_datasets[0] has three columns "Name", "Date" and "Count_Person". 

                # Sort the dataframe by Name and Date
                df_sorted = st.session_state.wen_datasets[0].sort_values(by=['Name', 'Date'])
                
                # Calculate the population difference using groupby and diff
                df_sorted['Count_Person_Diff'] = df_sorted.groupby('Name')['Count_Person'].diff()
                
                # Find the year with the greatest increment for each county
                max_increment_years = df_sorted.loc[df_sorted.groupby('yName')['Count_Person_Diff'].idxmax()]
                
                # Select only relevant columns
                st.session_state.wen_tables[0] = max_increment_years[['Name', 'Date', 'Count_Person_Diff']]

            For a column other than "Count_Person", please update the code accordingly.
            
            User question:
            {question}

            Answer:<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """,
        input_variables=["index", "title", "columns", "sample", "question"],
    )

    df_code_chain = prompt | llm | JsonOutputParser()
    df_code_chain2 = prompt | llm2 | JsonOutputParser()
    
    sample_df = st.session_state.wen_datasets[index].head(5)
    csv_string = sample_df.to_csv(index=False)

    try:
        return df_code_chain.invoke({'index': index,
                                     'title': st.session_state.wen_datasets[index].title,
                                     'columns': str(st.session_state.wen_datasets[index].dtypes),
                                     'sample': csv_string,
                                     'question': user_input})
    except Exception as e:
        st.markdown(str(e)) 
        return df_code_chain2.invoke({'index': index,
                                     'title': st.session_state.wen_datasets[index].title,
                                     'columns': str(st.session_state.wen_datasets[index].dtypes),
                                     'sample': csv_string,
                                     'question': user_input})


def remove_suffixes(place_name):
    # Define the pattern to match the suffixes "County", "State", or "City"
    pattern = r'\b(County|State|City)\b'
    
    # Use re.sub to remove the matched suffixes
    cleaned_name = re.sub(pattern, '', place_name).strip()
    
    # Optionally, remove any extra whitespace
    cleaned_name = re.sub(r'\s+', ' ', cleaned_name)
    
    return cleaned_name


def create_new_geodataframe(gdfs, df):
    # Create a dictionary to store geometries with "Name" as the key
    geometry_dict = {}
    
    # Iterate through each GeoDataFrame in the list
    for gdf in gdfs:
        for idx, row in gdf.iterrows():
            name = row['Name']
            geometry = row['geometry']
            geometry_dict[name] = geometry
    # st.code(geometry_dict.keys())
    
    # Initialize a list to store geometries for the new GeoDataFrame
    geometries = []
    
    # Iterate through the DataFrame df to build the new GeoDataFrame
    for idx, row in df.iterrows():
        name = row['Name']
        found = False
        for tmp in geometry_dict.keys():
            if name == tmp or remove_suffixes(name) == tmp:
                geometries.append(geometry_dict[tmp])
                found = True
                break
        if not found:
            raise ValueError(f"Geometry not found for name: {name}")
            
    # Create the new GeoDataFrame
    new_gdf = gpd.GeoDataFrame(df.copy(), geometry=geometries)
    
    return new_gdf

def strip_code(code):
    if code.startswith("```python"):
        start_index = code.find("```python") + len("```python")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    elif code.startswith("```"):
        start_index = code.find("```") + len("```")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    return code

def strip_json(code):
    if code.startswith("```json"):
        start_index = code.find("```json") + len("```json")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    elif code.startswith("```"):
        start_index = code.find("```") + len("```")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    return code

def strip_sparql(code):
    if code.startswith("```sparql"):
        start_index = code.find("```sparql") + len("```sparql")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    elif code.startswith("```"):
        start_index = code.find("```") + len("```")
        end_index = code.find("```", start_index)
        code = code[start_index:end_index].strip()
    elif code.startswith('"') and code.endswith('"'):
        code = code[1:-1]
    return code
    
def normalize_query_plan(data):
    for i in range(1, len(data)):
        # Check if the current item has 'WEN-OKN Database' and the previous has 'Energy Atlas'
        if data[i]["data_source"] == "WEN-OKN Database" and data[i-1]["data_source"] == "Energy Atlas":
            # Update the current item's data_source
            data[i]["data_source"] = "WEN-KEN database use Energy Atlas"
    return data

def spatial_dataset_exists(llm, request, spatial_datasets):
    if len(spatial_datasets) == 0:
         return { 'existing': False }
         
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

        [ Available Data ]
        The following are the variables with the data:
            {variables}
                        
        [ Question ]
        The following is the requested from the user:
            {question}

       Checks whether the user's current request is semantically fully equivalent to the processed 
       request of a geodataframe contained in a certain variable. Return a valid Python JSON string
       with a boolean field 'existing' to indicate if it exists and a string field "reason" to give 
       an explanation. Please return JSON only without any preamble or other explanation. 

       Note: Please note that “Find San Diego County” is not equivalent to “Find Southern San Diego County”.
       
       Note: Please note that "Find Scioto River" is NOT semantically equivalent to the processed request 
       "All basins that Scioto River flows through" because "Find Scioto River" tries to find a river with 
       the name "Scioto" and "All basins that Scioto River flows through" tries to find all basins that the
       Scioto River flows through".

       Note: Please note that "Find the tracts of all power stations at risk of flooding in Ohio at 2 PM on 
       July 1, 2025" is NOT semantically equivalent to the processed request "Find all power stations at risk 
       of flooding in Ohio that at 2 PM on July 1, 2025" because the first request tries to find 
       some census tracts and the second request tries to find some power stations at risk of flooding.

       Note: Please note that "Find the populations of the tracts of all buildings at risk of flooding in Ohio 
       at 2 PM on July 1, 2025" is NOT semantically equivalent to the processed request "Find the tracts
       of all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025" because the first request
       tries to find the populations of some tracts and the second request tries to find some tracts.

       In general, the following requests are not sematically equivalent:
            "Find the power stations .... ", 
            "Find the tracts of ....", 
            "Find the populations of ..."
        because they request different objects.
       
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "variables"],
    )
    df_code_chain = prompt | llm | JsonOutputParser()
 
    variables = ""
    if spatial_datasets:
        for index, dataset in enumerate(spatial_datasets):
            variables += f"""
                 st.session_state.datasets[{index}] holds a geodataframe after processing 
                 the request: { st.session_state.datasets[index].label}                                
                          """
    # st.code(variables)
    return df_code_chain.invoke({"question": request, "variables": variables})

def nonspatial_dataset_exists(llm, request, nonspatial_datasets):
    if len(nonspatial_datasets) == 0:
         return { 'existing': False }

    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

        [ Available Data ]
        The following are the variables with the data:
            {variables}
                        
        [ Question ]
        The following is the requested from the user:
            {question}

        Please check whether a varable contains the user's requested data. Return a valid Python JSON string
        with a boolean field 'existing' to indicate if it exists. Please return JSON only 
        without any explanations.  without preamble or explanation. 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "variables"],
    )
    df_code_chain = prompt | llm | JsonOutputParser()
 
    variables = ""
    if nonspatial_datasets:
        for index, dataset in enumerate(nonspatial_datasets):
            variables += f"""
                 st.session_state.wen_datasets[{index}] holds a dataframe after processing 
                 the request: { st.session_state.wen_datasets[index].title}                                
                          """
    # st.code(variables)
    return df_code_chain.invoke({"question": request, "variables": variables})

def normal_print(sparql):
    iow_graph = "GRAPH <http://iow.org>"
    iow_service = "SERVICE <https://graph.geoconnex.us/repositories/iow>"
    sparql = sparql.replace(iow_graph, iow_service)

    kwg_graph = "GRAPH <http://kwg.org>"
    kwg_service = "SERVICE <https://stko-kwg.geog.ucsb.edu/graphdb/repositories/KWG>"
    sparql = sparql.replace(kwg_graph, kwg_service)
    
    return sparql

def get_gdf_from_data_request(message, chat_container):
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner(f"""We're currently processing your request:
                                **{message}{'' if message.endswith('.') else '.'}**
                          Depending on the complexity of the query and the volume of data, 
                          this may take a moment. We appreciate your patience."""):

                # generate a sparql query. try up to 8 times because of the LLM limit
                max_tries = 8
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
                        elif data.startswith("```sql"):
                            start_index = data.find("```sql") + len("```sql")
                            end_index = data.find("```", start_index)
                            sparql_query = data[start_index:end_index].strip()
                        elif data.startswith("sql"):
                            start_index = data.find("sql") + len("sql")
                            sparql_query = data[start_index:].strip()
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
                        st.code(normal_print(sparql_query))

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
                        if gdf.shape[0] > 0:
                            return gdf
                    except Exception as e:
                        return None
                return None


def detect_4326_in_3857(gdf):
    """
    Detect if a geodataframe with CRS 3857 actually contains coordinates in 4326
    
    Args:
        gdf: GeoPandas DataFrame to check
        
    Returns:
        bool: True if likely contains 4326 coords, False otherwise
    """
    # Check if the declared CRS is 3857
    if gdf.crs is None or '3857' not in str(gdf.crs):
        print("GeoDataFrame CRS is not 3857")
        return False
        
    # Extract a sample of coordinates for checking
    x_coords = []
    y_coords = []
    
    # Get coordinates from first 100 geometries or all if less than 100
    sample_size = min(100, len(gdf))
    for i in range(sample_size):
        geom = gdf.geometry.iloc[i]
        if hasattr(geom, 'geoms'):  # For multigeometries
            for part in geom.geoms:
                x, y = part.representative_point().x, part.representative_point().y
                x_coords.append(x)
                y_coords.append(y)
        else:
            x, y = geom.representative_point().x, geom.representative_point().y
            x_coords.append(x)
            y_coords.append(y)
    
    # Check coordinate ranges
    # EPSG:4326 typically has values between -180 to 180 for longitude and -90 to 90 for latitude
    # EPSG:3857 typically has values in millions
    x_range = max(x_coords) - min(x_coords)
    y_range = max(y_coords) - min(y_coords)
    
    # Typical WebMercator (3857) coords are measured in meters and in the millions
    # If values are small, they're likely in degrees (4326)
    is_likely_4326 = (abs(max(x_coords)) <= 180 and abs(min(x_coords)) <= 180 and 
                      abs(max(y_coords)) <= 90 and abs(min(y_coords)) <= 90)
    
    if is_likely_4326:
        print("DETECTED: Your geodataframe claims to be in EPSG:3857 but coordinates appear to be in EPSG:4326")
        print(f"X range: {min(x_coords)} to {max(x_coords)}")
        print(f"Y range: {min(y_coords)} to {max(y_coords)}")
        print("To fix this issue, use: gdf = gdf.set_crs(epsg=4326, allow_override=True)")
        if y_range > 90 or x_range > 360:
            print("WARNING: While coordinates are in the range of EPSG:4326, some values exceed normal bounds")
    else:
        print("Coordinates appear to match the declared CRS:3857")
        
    return is_likely_4326
