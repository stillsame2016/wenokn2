
import requests
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

#####################################################################
# Implement the Aggregation Planer
def get_aggregation_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

You are an expert of following systems:
               1. The WEN-OKN knowledge database 
               2. Data Commons
               3. US Energy Atlas

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
        
You are also an expert in query analysis. Extract key components from the given user request, which describes an aggregation query.

Extraction Rules
    - Grouping Object: The entity used for grouping (e.g., county, state).
        * If not explicitly stated, infer the most reasonable entity from the query.
        * If multiple grouping entities exist, choose the most specific one.
    - Summarizing Object: The entity being aggregated (e.g., river, hospital).
        * If not explicitly stated, infer the entity that is being counted, summed, or aggregated.
        * Never use "aggregation" as a placeholder—always extract a meaningful entity.
    - Association Conditions: The relationship between the grouping and summarizing objects.
        * If missing, infer a reasonable relationship (e.g., "river flows through county").
    - Aggregation Function: The mathematical/statistical operation applied (e.g., COUNT, SUM, ARGMAX).
        * Always return in uppercase.
        * If missing, infer the most logical function based on the query.
    - Preconditions: Filters applied before aggregation (e.g., "county is in Ohio").
        * If none exist, return null.
    - Postconditions: Filters applied after aggregation (e.g., "COUNT > 5").
        * If none exist, return null.

Also please create a query plan which first load grouping objects by using preconditions and then load 
summarizing objects with proper bounding box and finally solve the request.

Example 1
User Request: "For each county in Ohio, find the number of rivers flowing through the county."

This request can be defined as the following query:
    SELECT county, COUNT(river) AS river_count   
    FROM county, river
    WHERE county in 'Ohio'  
      AND river INTERSECTS county  
    GROUP BY county
    
The object used in "GROUP BY" is the grouping object. The object used to apply the aggregation function COUNT is the summarizing object.

Extraction Output:
{{
  "grouping_object": "county",
  "summarizing_object": "river",
  "association_conditions": "river flows through county",
  "aggregation_function": "COUNT",
  "preconditions": "county in Ohio state",
  "postconditions": null,
  "query_plan": [
      {{ "request": "Find all counties in Ohio state",  "data_source": "WEN-OKN database"}}
      {{ "request": "Find all rivers intersects the proper bounding box", "data_source": "WEN-OKN database"}}
      {{ "request": "Find the number of rivers flowing through each county in Ohio state",  "data_source": "System"}}
  ]
}}

Strict Guidelines for Extraction
    - Do not return generic placeholders like "aggregation".
    - Ensure that "grouping_object" and "summarizing_object" are never null.
    - If the user request is ambiguous, infer the most logical structure.
    - Only return a JSON object. No explanations, no additional text.
    - For association conditions, construct a meaningful relationship between the grouping and summarizing objects.

User Request:
{question}

         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result

def get_code_for_grouping_object(llm, request):
    if request["data_source"] == "WEN-OKN database":
        response = requests.get(f"https://sparcal.sdsc.edu/api/v1/Utility/wenokn_llama3?query_text={request['request']}")
        sparql_query = response.text
        code = """
            endpoint = f"http://132.249.238.155/repositories/wenokn_ohio_all"
            df = sparql_dataframe.get(endpoint, '''{sparql_query}''')
            gdf = df_to_gdf(df, {request['request']})
        """
        return code
    return "OKAY"
