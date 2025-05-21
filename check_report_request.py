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

        Here are some sample queries and you can create more queries as need:
            'Show Ross County in Ohio State.', 
            'Show all counties in Kentucky State.', 
            'Find all counties the Scioto River flows through.',
            'Find all counties downstream of Ross County on the Scioto River.',  
            'Find all counties both the Ohio River and the Muskingum River flow through.',  
            'Find all counties downstream of the coal mine with the name Century Mine along Ohio River.',
            'Find all neighboring counties of Guernsey County.',
            'Find all adjacent states to the state of Ohio.',
            'Show the Ohio River.', 
            'Find all rivers that flow through Ross County.', 
            'What rivers flow through Dane County in Wisconsin?', 
            'Show all stream gauges on Muskingum River', 
            'Show all stream gages in Ross county in Ohio',
            'What stream gages are on the Yahara River in Madison, WI?',  
            'Find all stream gages on the Yahara River, which are not in Madison, WI',
            'Find all dams on the Ohio River.', 
            'Find all dams in Kentucky State.',
            'Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river',
            'Show the populations for all counties in Ohio State.', 
            'Find populations for all adjacent states to the state of Ohio.',
            'Find the median individual income for Ross County and Scioto County.', 
            'Find the number of people employed in all counties the Scioto River flows through.', 
            "Show social vulnerability index of all counties downstream of coal mine with the name 'Century Mine' along Ohio River",
            'Find all solar power plants in California.', 
            'Find all coal mines along the Ohio River.', 
            'Where are the coal-fired power plants in Kentucky?',
            'Show natural gas power plants in Florida.',
            'Load all wind power plants with total megawatt capacity greater than 100 in California.' ,
            'Find the basin Lower Ohio-Salt',
            'Find all basins through which the Scioto River flows.',
            'Find all rivers that flow through the Roanoke basin.',
            'Find all watersheds in the Kanawha basin.',  
            'Find all watersheds feed into Muskingum River',
            'Find all watersheds in Ross County in Ohio State',
            'Find the watershed with the name Headwaters Black Fork Mohican River',
            'Find all stream gages in the watershed with the name Meigs Creek',
            'Find all stream gages in the watersheds feed into Scioto River',
            'Find all rivers that flow through the watershed with the name Headwaters Auglaize River'      

        User Request: {question} 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result
