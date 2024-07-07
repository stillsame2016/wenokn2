import geopandas as gpd
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def process_wenokn_use_energy_atlas(llm, user_input):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

        We have two data query systems: WEN-OKN database and Energy Atlas.

        The WEN-KEN database contains following entities: 
          1. Locations: Information on buildings, power stations, and underground storage tanks in Ohio.
          2. Counties: Geometric representations of counties across the USA.
          3. States: Geometric representations outlining the boundaries of states in the USA.
          4. Earthquakes: Data pertaining to seismic events.
          5. Rivers: Comprehensive geometries about rivers in USA.
          6. Dams: Information regarding dams' locations in USA.
          7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
          8. Hospitals: Details about hospital locations and information in USA.

        The following is the description of Energy Atlas:

        [ Definition 1 for Energy Atlas ] 
        We have the following function to get coal mines from an ArcGIS Feature Service as a 
        GeoDataFrame:
            load_coal_mines(where_condition)
        
        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'MSHA_ID', 'MINE_NAME', 'MINE_TYPE',
            'MINE_STATE', 'STATE', 'FIPS_COUNTY', 'MINE_COUNTY', 'PRODUCTION',
            'PHYSICAL_UNIT', 'REFUSE', 'Source', 'PERIOD', 'Longitude', 'Latitude'
            
        Use the column 'STATE' rather than the column 'MINE_STATE' to find coal mines in a state. 
        The values in the column 'STATE' are all in upper case like 'ALABAMA' or 'COLORADO' etc. 
        The column 'COUNTY' contains values like 'Walker' or 'Jefferson'. 
        
        To get all coal mines, call load_coal_mines with "1 = 1" as where_condition.

        [ Definition 2 for Energy Atlas ] 
        We have the following functions to get coal power plants/wind power plants/battery storage plants/
        geothermal power plants/hydro pumped storage power plants/natural gas power plants/nuclear power plants/
        petroleum power plants/solar power plants from an ArcGIS Feature Service as a GeoDataFrame:
            load_coal_power_plants(where_condition)
            load_wind_power_plants(where_condition)
            load_battery_storage_plants(where_condition)
            load_geothermal_power_plants(where_condition)
            load_hydro_pumped_storage_power_plants(where_condition)
            load_natural_gas_power_plants(where_condition)
            load_nuclear_power_plants(where_condition)
            load_petroleum_power_plants(where_condition)
            load_solar_power_plants(where_condition)
        
        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'Plant_Code', 'Plant_Name', 'Utility_ID', 'Utility_Name', 'sector_name', 
            'Street_Address', 'City', 'County', 'State', 'Zip', 'PrimSource', 'source_desc', 'tech_desc', 
            'Install_MW', 'Total_MW', 'Bat_MW', 'Bio_MW', 'Coal_MW', 'Geo_MW', 'Hydro_MW', 'HydroPS_MW', 
            'NG_MW', 'Nuclear_MW', 'Crude_MW', 'Solar_MW', 'Wind_MW', 'Other_MW', 'Source', 'Period', 
            'Longitude', 'Latitude'
            
        The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' etc. 
        The column 'County' contains values like 'Adams' or 'Yellowstone'. The column 'Total_MW' gives the 
        Total Megawatts of the plants.

        [ Definition 3 for Energy Atlas ]
        We have the following function to get renewable diesel fuel and other biofuel plants/biodiesel plants
        from an ArcGIS Feature Service as a GeoDataFrame:
            load_renewable_diesel_fuel_and_other_biofuel_plants(where_condition)
            load_biodiesel_plants(where_condition)

        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'Company', 'Site', 'State', 'PADD', 'Cap_Mmgal',
           'Source', 'Period', 'Longitude', 'Latitude'
           
        The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' etc.
        
        To get all coal mines/coal power plants/wind power plants/renewable diesel fuel and 
        other biofuel plants and etc, call the correspondent function with "1 = 1" as where_condition.
        

        The following is the user request:
        {question}

        This request asks for entities from WEN-OKN database that satisfy certain conditions, some of 
        which are described using entities in Energy Atlas. Your task is to provide Python code which 
        converts this request into a request in natural language without using any conditions related 
        to energy Atlas as a Python variable converted_request. Please return the Python code only 
        without any explanation. Don't include any print statement. Don't add ``` around the code.

        [ Example ]
        Find all counties downstream of the coal mine with the name "Century Mine" along Ohio River.

        First we find the latitude and longitude of the coal mine with the name "Century Mine". Then convert the original 
        request into a request without using Energy Atlas. The following is returned Python code:

            gdf =  load_coal_mines("MINE_NAME = 'Century Mine'")
            # Keep this line for debug
            st.code(gdf.shape)
            row = gdf.iloc[0]
            latitude = row['LATITUDE']
            longitude = row['LONGITUDE']
            converted_request = f"Find all counties downstream of the point at the latitude {{latitude}} and longitude {{longitude}} along Ohio River"
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    df_code_chain = prompt | llm | StrOutputParser()
    return df_code_chain.invoke({"question": user_input})

