from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
def check_report_request(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Check the question and return a JSON string in the
        following format:
        {{
            "create_report": true/false
        }}

        Set "create_report": true if the question implies intent to generate, produce, or create a report, summary, or document. Look for phrases like:
           - "generate a report"
           - "create a summary"
           - "compile a report"
           - "produce findings"
           - "document the results"

        Otherwise, set "create_report": false.

        Question to check: {question} 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result

#####################################################################
def create_report_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        Our information system has the following data:

        WEN-KEN database contains the following entities:
        1. Locations: Information on buildings, power stations, and underground storage tanks in Ohio.
        2. Counties: Geometric representations of counties across the USA.
        3. States: Geometric representations outlining the boundaries of states in the USA.
        4. Earthquakes: Data pertaining to seismic events.
        5. Rivers: Comprehensive geometries about rivers in USA.
        6. Dams: Information regarding dams' locations in USA.
        7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
        8. Hospitals: Details about hospital locations and information in USA.
        9. Stream Gages: Information of gages' locations and names in USA.
        10. Power Plants: Information about power plants in Ohio.
        You do not need to be stringent with the keywords in the question related to these topics.
        
        Data Commons contains following data for counties or states or countries or FIPS.
            Area_FloodEvent
            Count_Person (for population)
            Count_FireEvent
            Count_FlashFloodEvent
            Count_FloodEvent
            Count_HailEvent
            Count_HeatTemperatureEvent
            Count_HeatWaveEvent
            Count_HeavyRainEvent
            Count_Person_Employed
            Median_Income_Person
            CountOfClaims_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
            Max_Rainfall
            Max_Snowfall
            SettlementAmount_NaturalHazardInsurance_BuildingContents_FloodEvent
            SettlementAmount_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
            SettlementAmount_NaturalHazardInsurance_BuildingStructure_FloodEvent
            FemaSocialVulnerability_NaturalHazardImpact
        
        US Energy Atlas contains following data:
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
            Watershed
            Basin
        
        We need to generate a report based on the user request. The report must be based on our existing data and cannot refer to 
        any other data that does not appear in our existing data.
        
        You need to generate a series of queries written in natural language that will be used to get the data for the report. 
        We will run these queries and then send you the query results to generate a report based on the data obtained. Please return 
        a JSON string in the following format:
            [ "query 1", "query 2", ... ]

        User Request: {question} 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result
