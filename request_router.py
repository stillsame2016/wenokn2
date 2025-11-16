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
            Basin
	        Census Block
            Census Tract
	        Power Station at Risk of Flooding
            Building at Risk of Flooding
            PFAS Contamination Observation
			Public Water System
			USDA ARS site
			USDA ARS site with pesticide
			Military Base
			FRS Facility in Maine or Ohio or Illinois (including 
				Waste Treatment and Disposal
				Converted Paper Manufacturing
				Water Supply and Irrigation
				Sewage Treatment
				Plastics Product Manufacturing
				Textile and Fabric Finishing and Coating
				Basic Chemical Manufacturing
				Paint, Coating, and Adhesive Manufacturing
				Aerospace Product and Parts
				Drycleaning and Laundry Services
				Carpet and Upholstery Cleaning Services)
			
        Use "WEN-KEN database use Energy Atlas" for the requests to find entities from WEN-KEN database but with
        somes join condition for the entities from WEN-KEN database and Energy Atlas.

        Note that use ""WEN-KEN database" for power stations and use "US Energy Atlas" for power plants.

 	Use "US Energy Atlas" for buildings or power stations at risk of flooding at a time in some states or counties.

        All the rules above don't work, then conside use "Aggregation" for questions related to aggregation requests, which compute summaries over grouped entities.
        An aggregation request may involve 5 core components:
            1) Grouping Objects: Entities to partition data by (e.g., counties, basins).
            2) Summarizing Objects: Entities to aggregate (e.g., rivers, dams).
            3) Association Conditions: Relationships between grouping and summarizing objects (e.g., spatial containment, spatail intersection).
            4) Aggregation Function: Operations like COUNT, SUM, MAX, AVG, or ARGMAX (for object-centric results).
            5) Pre-/Post-Conditions: Filters applied before/after aggregation (e.g., counties in Ohio State, result thresholds).
        Please note that an aggregation request must use an aggregation function. It is not an aggregation request if no aggregation function is used.
	For example, "find all counties Scioto River flows through" is not an aggregation request because it doesn't use any aggregation function.
	
        [ Example 1 ]
        Return 'WEN-KEN database' for following requests: 
		    Find all neighboring states of Ohio State.
			Find all counties both the Ohio River and the Muskingum River flow through.
			Find all rivers passes the counties Scioto River passes.
			Find all downstream counties of the Scioto River from Ross County.
			Find all dams in all counties that the Ohio River passes.
		Because counties, rivers and dams are contained in WEN-KEN database.

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
        Because this request tries to find coal mines which is contained in Energy Atlas.

        [ Example 5 ]
        Return "Data Commons" for the following request:
            Find the social vulnerability for all counties downstream of the coal mine with the name "Century Mine" along Ohio River
        Because this request tries to find the social vulnerability of some counties which is contained in Data Commons.

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
        Return 'WEN-KEN database' for following request: 
            Find all dams located upstream of the power station dpjc6wtthc32 along the Muskingum river.
        Because this request tried to find dams which are contained in the WEN-KEN database.

        [ Example 9 ]
        Return 'WEN-KEN database' for following requests: 
            Find the Ohio State 
        Because this request tries to find a state which is contained in the WEN-KEN database.

        [ Example 10 ]
        Return "WEN-KEN database use Energy Atlas" for the following request: 
            Find all rivers that flow through the Roanoke basin. 
        Because this request tries to find some rivers (in WEN-KEN database) but with some conditions related to the Roanoke basin in Energy Atlas.

        [ Example 11 ]
        Return "Energy Atlas" for the following request:
            Find all basins through which the Scioto River flows. 
        Because this request tried to find some basins which are contained in Energy Atlas.

        [ Example 12 ]
        Return "Energy Atlas" for the following request:
            Find all basins that intersect with Ohio River. 
        Because this request tried to find some basins which are contained in Energy Atlas.

        [ Example 12 ]
        Return "Energy Atlas" for the following request:
            Find the tracts of power stations at risk of flooding in Ohio at 2 PM on July 1, 2025.
        Because this request tried to find some consus tracts which are contained in Energy Atlas, and all
	    power stations in Ohio that are risk of flooding at 2 PM on July 1, 2025 are also contained in Energy Atlas.

        [ Example 13 ]
        Return "Data Commons" for the following requests:
            Find the number of people employed in all counties the Scioto River flows through. 
	    Find social vulnerability index of the census tract for the power station dnuzf75z9hdz.

        [ Example 14]
        Return "Aggregation" for the following requests:
            Find the number of rivers flow through each county in Ohio. (use COUNT and group by county)
            Find the number of dams in each county in Ohio.  (use COUNT and group by county)
            Find the total number of coal mines in each basin. (use COUNT and group by basin)
            Find the total power generation capacity of gas power plants in each county in Ohio. ((use SUM and group by county)
            Find the longest river in county of Ohio. (use MAX and group by county)
            Find the county with the the highest number of hospitals in Ohio .  (use MAX)
	        Find all counties with more than 5 hospitals in Ohio . (use COUNT and group by county)
	        Find all states where the total coal mine output exceeds 1 million tons. (use SUM and group by state)
            Find the river in Ohio that has the highest number of dams. (use MAX adn COUNT and group by river)
	        Find the watershed that has the highest total coal mine. (use MAX and COUNT and group by watershed)

        [ Example 15 ]
        Return "Energy Atlas" for the following request:
            Find PFSA contamination observations within 100 meters to Presumpscot River
        Because this request tried to find PFSA contamination observations which are contained in Energy Atlas.
		
        Question to route: {question} 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result
