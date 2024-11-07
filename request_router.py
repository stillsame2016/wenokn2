from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
# Implement the Router
def get_question_route(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to WEN-KEN database or NPDES regulations or Data Commons or US Energy Atlas. 
        
        Use the WEN-KEN database for questions on the following entities: 
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
        
        Use NPDES regulation for questions related to permits or permit applications of discharges pollutants into 
        navigable waters, which include rivers, lakes, streams, coastal areas, and other bodies of water. Point 
        sources are discrete conveyances such as pipes, ditches, or channels. Under the NPDES program, permits are 
        issued to regulate the quantity, quality, and timing of the pollutants discharged into water bodies. These 
        permits include limits on the types and amounts of pollutants that can be discharged, monitoring and 
        reporting requirements, and other conditions to ensure compliance with water quality standards and protect 
        the environment and public health.

        Use Data Commons for questions related to the following data for counties or states or countries. 
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
        
        For example, to find the populations of all counties where Muskingum River flows through, we need to find 
        the populations for the counties satisfying some conditions. In this case, this request uses "Data Commons".

        Use US Energy Atlas for requests related to the following data:
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

        Use "WEN-KEN database use Energy Atlas" for the requests to find entities from WEN-KEN database but with
        somes join condition for the entities from WEN-KEN database and Energy Atlas.

        Note that use ""WEN-KEN database" for power stations and use "US Energy Atlas" for power plants.

        [ Example 1 ]
        Return 'WEN-KEN database' for following request: Find all neighboring states of Ohio State.

        [ Example 2 ]
        Return "WEN-KEN database use Energy Atlas" for the following request: 
            Find counties downstream of the coal mine with the name "Century Mine" on the Ohio River. 
        Because this request tries to find some counties (in WEN-KEN database) but with some conditions related to
        Ohio River (in WEN-KEN database) and the coal mine with the name "Century Mine" in Energy Atlas.

        [ Example 3 ]
        Return "Energy Atlas" for the following request:
            Find all coal mines along Ohio River
        Because this request tries to find coal mines (in Energy Atlas rather than in WEN-KEN database)

        [ Example 4 ]
        Return "Energy Atlas" for the following request:
            Find all coal mines in all counties the Scioto River flows through.
        Because this request tries to find coal mines (in Energy Atlas rather than in WEN-KEN database)

        [ Example 5 ]
        Return "Data Commons" for the following request:
            Find the social vulnerability for all counties downstream of the coal mine with the name "Century Mine" along Ohio River
        Because this request tries to find the social vulnerability of some counties which satisfy some conditions.

        Use "Other" for questions related to common knowledge. 

        [ Example 6 ]
        Return "Other" for the following request:
            Please help search Kentucky Public Service Commission's website to find out how many power plants are in Kentucky
        
        Give a choice 'WEN-KEN database' or 'NPDES regulations' or 'Data Commons' or 'US Energy Atlas' or 
        'WEN-KEN database use Energy Atlas' or 'Other' based on the question. Return a JSON with a single key 
        'request_type' and a key 'explanation' for reasons. 

        [ Example 7 ]
        Return the following JSON string for the request "Could you please give me some example questions?":
            {{
              "request_type" : "Other",
              "explanation" : "Not in the scope of 'WEN-KEN database' or 'NPDES regulations' or 'Data Commons' or 'US Energy Atlas' or 'WEN-KEN database use Energy Atlas'"
            }}

        [ Example 8 ]
        Return 'WEN-KEN database' for following request: Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river.
        The WEN-KEN database contains power stations and US Energy Atlas contains power plants.

        [ Example 9 ]
        Return 'WEN-KEN database' for following request: Find the Ohio State. The WEN-KEN database contains all USA states and geometries.

        Question to route: {question} 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result
