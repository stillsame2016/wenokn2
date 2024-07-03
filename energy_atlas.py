import geopandas as gpd
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def process_energy_atlas_request(llm, user_input, spatial_datasets):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

        [ Definition 1 ] 
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

        [ Definition 2 ] 
        We have the following functions to get coal power plants/wind power plants/battery storage plants/
        geothermal power plants/hydro pumped storage power plants/natural gas power plants/nuclear power plants/
        petroleum power plants/solar power plants/load_biodiesel_plant from an ArcGIS Feature Service as a GeoDataFrame:
            load_coal_power_plants(where_condition)
            load_wind_power_plants(where_condition)
            load_battery_storage_plant(where_condition)
            load_geothermal_power_plant(where_condition)
            load_hydro_pumped_storage_power_plant(where_condition)
            load_natural_gas_power_plant
            load_nuclear_power_plant
            load_petroleum_power_plant
            load_solar_power_plant
            load_biodiesel_plant
        
        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'Plant_Code', 'Plant_Name', 'Utility_ID', 'Utility_Name', 'sector_name', 
            'Street_Address', 'City', 'County', 'State', 'Zip', 'PrimSource', 'source_desc', 'tech_desc', 
            'Install_MW', 'Total_MW', 'Bat_MW', 'Bio_MW', 'Coal_MW', 'Geo_MW', 'Hydro_MW', 'HydroPS_MW', 
            'NG_MW', 'Nuclear_MW', 'Crude_MW', 'Solar_MW', 'Wind_MW', 'Other_MW', 'Source', 'Period', 
            'Longitude', 'Latitude'
            
        The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' etc. 
        The column 'County' contains values like 'Adams' or 'Yellowstone'. 

        [ Definition 3 ]
        We have the following function to get renewable diesel fuel and other biofuel plants 
        from an ArcGIS Feature Service as a GeoDataFrame:
            load_renewable_diesel_fuel_and_other_biofuel_plants(where_condition)

        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'Company', 'Site', 'State', 'PADD', 'Cap_Mmgal',
           'Source', 'Period', 'Longitude', 'Latitude'
           
        The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' etc.
        
        To get all coal mines/coal power plants/wind power plants/renewable diesel fuel and 
        other biofuel plants and etc, call the correspondent function with "1 = 1" as where_condition.

        [ Available Data ]
        The following are the variables with the data:
            {variables}
                        
        [ Question ]
        The following is the question from the user:
            {question}

        Please return only the complete Python code in the following format to implement the user's request without preamble or 
        explanation. 
            # other code which do not change spatial_datasets
            gdf = ......
            gdf.title = ......
        Don't include any print statement. Don't add ``` around the code. Make a title and save the title in gdf.title.  

        [ Example 1]
        Find all coal mines along Ohio River. 

        Assume gdf1 contains Ohio River only. Then you can return the following code:
            gdf2 = load_coal_mines("1 = 1")
            # Keep the following line exactly as it is
            distance_threshold = 0.2
            gdf2['distance_to_river'] = gdf2.geometry.apply(lambda x: gdf1.distance(x).min())
            gdf = gdf2[gdf2['distance_to_river'] <= distance_threshold]
            gdf = gdf.drop(columns=['distance_to_river'])
            gdf.title = "All Coal Mines within 10 Miles away from Ohio River"

        [ Example 2 ]
        Find all coal power plants along Ohio River.

        Use the same way as Example 1 to implement it. Just replace load_coal_mines by load_coal_power_plants
        and change the title.

        [ Note 1 ]
        Use pandas.concat to concatenate two geodataframe gdf1 and gdf2:
            gdf = pd.concat([gdf1, gdf2], ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question", "variables"],
    )
    df_code_chain = prompt | llm | StrOutputParser()
 
    variables = ""
    if spatial_datasets:
        for index, dataset in enumerate(spatial_datasets):
            variables += f"""
                             st.session_state.datasets[{index}] holds a geodataframe after processing 
                             the request: { st.session_state.datasets[index].label}
                             The following is the columns of st.session_state.datasets[{index}]:
                                 { st.session_state.datasets[index].dtypes }
                             The following is the first 5 rows of the data:
                                 { st.session_state.datasets[index].head(5).drop(columns='geometry').to_csv(index=False) }
                                 
                          """
    # st.code(variables)
    return df_code_chain.invoke({"question": user_input, "variables": variables})


def load_features(self_url, where, wkid):
    url_string = self_url + "/query?where={}&returnGeometry=true&outFields={}&f=geojson".format(where, '*')
    resp = requests.get(url_string, verify=False)
    data = resp.json()
    if data['features']:
        return gpd.GeoDataFrame.from_features(data['features'], crs=f'EPSG:{wkid}')
    else:
        return gpd.GeoDataFrame(columns=['geometry'], crs=f'EPSG:{wkid}')

def load_coal_mines(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/CoalMines_US_EIA/FeatureServer/247"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_coal_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Coal_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_wind_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Wind_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_renewable_diesel_fuel_and_other_biofuel_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Renewable_Diesel_and_Other_Biofuels/FeatureServer/245"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_battery_storage_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Battery_Storage_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_geothermal_power_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Geothermal_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_hydro_pumped_storage_power_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Hydro_Pumped_Storage_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)    

def load_natural_gas_power_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Natural_Gas_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)   

def load_nuclear_power_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Nuclear_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_petroleum_power_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Petroleum_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_solar_power_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Solar_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_biodiesel_plant(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Biodiesel_Plants_US_EIA/FeatureServer/113"
    wkid = "3857"
    return load_features(self_url, where, wkid)  





