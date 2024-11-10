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
            
        The values in the column 'State' are case sensitive like 'Nebraska' or 'Montana' or 'Kentucky' etc. 
        The column 'County' contains values like 'Adams' or 'Yellowstone'. The column 'Total_MW' gives the 
        Total Megawatts of the plants.

        Note that use the case sensitive state names for the column 'State'.

        [ Definition 3 ]
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

        [ Definition 4 ]
        We have the following function to get watersheds from an ArcGIS Feature Service as a GeoDataFrame:
            load_watersheds(where_condition, bbox)
        where bbox is for a bounding box. Use None if bbox is unknown or not needed. 

        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'HUC10', 'NAME', 'HUTYPE', 'Shape__Area', 'Shape__Length'

        Use the following condition when trying to get a watershed by a given watershed name (e.g., Headwaters Scioto River):
            NAME LIKE '%Headwaters Scioto River%'
        The reason for this is that there may be spaces in the name column of the ArcGIS Feature service.
        
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

        Find out if one of the available variables is a geodataframe containing Ohio River.

        If none of the available variables are geodataframes containing Ohio River, then return the following code:
            raise Exception("The data for Ohio River is missing. Please load Ohio River first.")
        
        If you found a variable which is a geodataframe containing Ohio River, then return the valid Python code in the 
        following format:
            gdf1 = <replace by the variable of the geodataframe for Ohio River if you found one>
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
        and change the title. If none of the available variables are geodataframes containing Ohio River, then return the
        code raising the execption.

        [ Example 3 ]
        If the request is for an attribute of a particular plant, first obtain the plant as gdf, and then store the answer 
        to the user in gdf.answer. For example, find the capacity of the coal power plant Rockport.
            gdf = load_coal_power_plants("Plant_Name = 'Rockport'")
            gdf.title = "The Coal Power Plant Rockport"
            gdf.answer = f"The capacity of the coal power plant Rockport is {{gdf.iloc[0]['Total_MW']}} megawatt."

        [ Note 1 ]
        Use pandas.concat to concatenate two geodataframe gdf1 and gdf2:
            gdf = pd.concat([gdf1, gdf2], ignore_index=True)
            gdf = gpd.GeoDataFrame(gdf, geometry='geometry')

        [ Example 4 ]
        Find all solar power plants in all counties the Scioto River flows through.

        Find out if one of the available variables is a geodataframe containing all counties the Scioto River flows through.

        If none of the available variables are geodataframes containing all counties the Scioto River flows through, 
        then return the following code:
            raise Exception("The data for all counties the Scioto River flows through is missing. Please load it first.")
        
        If you found a variable which is a geodataframe containing all counties the Scioto River flows through, then return 
        the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for all counties the Scioto River flows through if you found one>
            gdf2 = load_solar_power_plants("1 = 1")
            gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
            gdf = gdf[gdf2.columns]
            gdf.title = "All solar power plants in all counties the Scioto River flows through"

        [ Example 5 ]
        Find all the watersheds that feed into the Scioto River.
        
        Find out if one of the available variables is a geodataframe containing Scioto River.

        If none of the available variables are geodataframes containing Scioto River, 
        then return the following code:
            raise Exception("The data for the Scioto River is missing. Please load it first.")
        
        If you found a variable which is a geodataframe containing Scioto River, then return 
        the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for the Scioto River if you found one>
            gdf2 = load_watersheds("1 = 1", gdf1.total_bounds)
            buffer_distance = 0.01
            gdf1_buffered = gdf1.copy()
            gdf1_buffered['geometry'] = gdf1_buffered['geometry'].buffer(buffer_distance)
            gdf = gpd.sjoin(gdf2, gdf1_buffered, how="inner", predicate="intersects")
            gdf = gdf[gdf2.columns]
            gdf.title = "All the watersheds that feed into the Scioto River"

        [ Example 6 ]
        Find all the watersheds in Ohio State.

        Find out if one of the available variables is a geodataframe containing Ohio State.

        If none of the available variables are geodataframes containing Ohio State, 
        then return the following code:
            raise Exception("The data for the Ohio State is missing. Please load it first.")
        
        If you found a variable which is a geodataframe containing Ohio State, then return 
        the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for the Ohio State if you found one>
            gdf2 = load_watersheds("1 = 1", gdf1.total_bounds)
            gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
            gdf = gdf[gdf2.columns]
            gdf.title = "All the watersheds in Ohio State"

        [ Example 7 ]
        Find all the watersheds in Ross County in Ohio State.

        Find out if one of the available variables is a geodataframe containing Ross County in Ohio State.

        If none of the available variables are geodataframes containing Ross County in Ohio State, 
        then return the following code:
            raise Exception("The data for Ross County in Ohio State is missing. Please load it first.")
        
        If you found a variable which is a geodataframe containing Ohio State, then return 
        the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for Ross County in Ohio State if you found one>
            gdf2 = load_watersheds("1 = 1", gdf1.total_bounds)
            gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
            gdf = gdf[gdf2.columns]
            gdf.title = "All the watersheds in Ross County in Ohio State"
        
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


def get_arcgis_features(self_url, where, bbox=None):
    if bbox is None:
        bbox = [-125.0, 24.396308, -66.93457, 49.384358]
    minx, miny, maxx, maxy = bbox
    params = {
        "where": where,
        "geometry": f"{minx},{miny},{maxx},{maxy}",
        "geometryType": "esriGeometryEnvelope",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "outSR": "4326",  # Ensure output is in WGS84                                                                                                                
        "resultOffset": 0,
        "resultRecordCount": 1000  # Increase this if needed                                                                                                         
    }

    response = requests.get(self_url + "/query", params=params)
    data = response.json()
    # st.code(response.url)
    # st.code(data)
    if data['features']:
        return gpd.GeoDataFrame.from_features(data['features'])
    else:
        return gpd.GeoDataFrame(columns=['geometry'])

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

def load_battery_storage_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Battery_Storage_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_geothermal_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Geothermal_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)

def load_hydro_pumped_storage_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Hydro_Pumped_Storage_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)    

def load_natural_gas_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Natural_Gas_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)   

def load_nuclear_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Nuclear_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_petroleum_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Petroleum_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_solar_power_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Solar_Power_Plants/FeatureServer/0"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_biodiesel_plants(where):
    self_url = "https://services7.arcgis.com/FGr1D95XCGALKXqM/ArcGIS/rest/services/Biodiesel_Plants_US_EIA/FeatureServer/113"
    wkid = "3857"
    return load_features(self_url, where, wkid)  

def load_watersheds(where, bbox):
    self_url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Watershed_Boundary_Dataset_HUC_10s/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, bbox)  



