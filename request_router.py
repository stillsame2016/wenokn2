from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
# Implement the Router
def get_question_route(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an expert at routing a 
        user question to WEN-KEN database or NPDES regulations or Data Commons. 
        
        Use the WEN-KEN database for questions on the following entities: 
          1. Locations: Information on buildings, power stations, and underground storage tanks in Ohio.
          2. Counties: Geometric representations of counties across the USA.
          3. States: Geometric representations outlining the boundaries of states in the USA.
          4. Earthquakes: Data pertaining to seismic events.
          5. Rivers: Comprehensive geometries about rivers in USA.
          6. Dams: Information regarding dams' locations in USA.
          7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
          8. Hospitals: Details about hospital locations and information in USA.
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
            CountOfClaims_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
            Max_Rainfall
            Max_Snowfall
            SettlementAmount_NaturalHazardInsurance_BuildingContents_FloodEvent
            SettlementAmount_NaturalHazardInsurance_BuildingStructureAndContents_FloodEvent
            SettlementAmount_NaturalHazardInsurance_BuildingStructure_FloodEvent
        
        For example, to find the populations of all counties where Muskingum River flows through, we need to find 
        the populations for the counties satisfying some conditions. In this case, this request uses "Data Commons".
        
        Use Other for questions related to common knowledge. 
        
        Give a choice 'WEN-KEN database' or 'NPDES regulations' or 'Data Commons' or 'Other' based on the question. 
        Return a JSON with a single key 'request_type' and a key 'explanation' for reasons. 
        
        Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result
