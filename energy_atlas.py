import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import concurrent.futures
import time
from typing import Optional, List, Dict, Any, Union
import logging

import io
import pyproj
import json
from shapely.geometry import Point, box, LineString, MultiLineString
from shapely.ops import transform
import shapely.geometry
import shapely.ops
import numpy as np

from shapely import wkt
import sparql_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

        [ Definition 5 ]
        We have the following function to get basins from an ArcGIS Feature Service as a GeoDataFrame:
            load_basins(where_condition, bbox)
        where bbox is for a bounding box. Use None if bbox is unknown or not needed. 

        The returned GeoDataFrame has the following columns:
            'geometry', 'OBJECTID', 'HUC6', 'NAME', 'Shape__Area', 'Shape__Length'

        Use the following condition when trying to get a watershed by a given watershed name (e.g., Headwaters Scioto River):
            UPPER(NAME) = UPPER('Headwaters Scioto River')
        The reason for this is that there may be spaces in the name column of the ArcGIS Feature service.

        [ Definition 6 ]
        We have the following function to get a Census Block from latitude and longitude from an ArcGIS Feature Service as a GeoDataFrame:
             load_census_block(latitude, longitude)
  
        We also have the following function to load all Census Blocks within a distance to the specified latitude/longitude as a GeoDataFrame.           
             load_nearby_census_blocks(latitude, longitude, radius_miles=5)
        
        Using load_nearby_census_blocks when the request tries to find all census blocks within a distance to some place. See the Example 9 below.

        The returned GeoDataFrame has the following columns:
                STATE:	2-digit FIPS code for the state (e.g., "06" = California).
                COUNTY:	3-digit FIPS code for the county (e.g., "073" = San Diego County).
                TRACT:	6-digit Census Tract code within the county (e.g., "008362").
                BLKGRP:	1-digit Block Group number within the tract (e.g., "3").
                BLOCK:	4-digit Census Block number (e.g., "3000").
                SUFFIX:	Block suffix (usually "Null"; rarely used to distinguish overlapping blocks).
                GEOID:	Full 15-digit identifier for the block: STATE + COUNTY + TRACT + BLOCK.
                LWBLKTYP:	Block type code (e.g., "L" = large area block like parks or unpopulated land).
                UR:	Urban/rural indicator: "U" = Urban, "R" = Rural.
                AREAWATER:	Area of water in square meters.
                AREALAND:	Area of land in square meters.
                MTFCC:	Feature classification code (e.g., "G5040" = census block).
                NAME:	Block name (e.g., "Block 3000").
                BASENAME:	Base block number without the "Block" prefix (e.g., "3000").
                LSADC:	Legal/statistical area description code: "BK" = Block.
                FUNCSTAT:	Functional status: "S" = Statistical (used for census tabulation).
                CENTLON:	Longitude of the polygon centroid.
                CENTLAT:	Latitude of the polygon centroid.
                INTPTLON:	Longitude of the internal point used for label placement.
                INTPTLAT:	Latitude of the internal point.
                HU100:	Number of housing units in the block (100% count from the 2020 Census).
                POP100:	Population count in the block (100% count from the 2020 Census).

        [ Definition 7 ]
        We have the following function to get a Census Tract from latitude and longitude from an ArcGIS Feature Service as a GeoDataFrame:
             load_census_tract(latitude, longitude)
        The returned GeoDataFrame has the following columns:
              'MTFCC', 'OID', 'GEOID', 'STATE', 'COUNTY', 'TRACT', 'BASENAME', 'NAME', 'LSADC', 
              'FUNCSTAT', 'AREALAND', 'AREAWATER', 'CENTLAT', 'CENTLON', 'INTPTLAT', 'INTPTLON', 'OBJECTID'

        We also have the following function to retrieve all Census Tracts located downstream of a river, starting from one or more specified 
        points represented as a GeoDataFrame. The function returns the downstream tracts as a GeoDataFrame.
              downstream_tracts(river_gdf, point_gdf, flow_dir='south')

        [ Definition 8 ]
        We have the following function to get power stations at risk of flooding in some state or county FIPS codes at an hour as a GeoDataFrame:
             load_flooded_power_stations(date, scope)
        where date in format YYYYMMDDHH (e.g., "2025071414") and scope is str or list of str, can be:
           - State FIPS code (e.g., "39" for Ohio)
           - County FIPS code (e.g., "39009" for Athens County, Ohio)
           - List of FIPS codes
        If None, defaults to Ohio ("39") and Kentucky ("21")

        The returned GeoDataFrame has the following columns:                                                                                                              
            'fips': FIPS code for tract,                                                                                                                                      
            'feature-type': the constant 'power',                                                                                                                                     
            'geometry': the tract polygon.    

        [ Definition 9 ]
        We have the following function to get buildings at risk of flooding in some state or county FIPS codes at an hour as a GeoDataFrame:
             load_flooded_buildings(date, scope)

        [ Definition 10 ]
        We have the following function to get all PFSA contamiation observations as a GeoDataFrame:
             load_PFAS_contamiation_observations()
        with the following columns: 'Obs', 'Substance', 'Date', 'Value', 'Unit', 'SamplePoint', 'geometry'.

        [ Definition 11 ]
        We have the following function to get all public water systems in a state as a GeoDataFrame:
             load_public_water_systems(state_name: str = "maine", limit: int = 2000)
        with the following columns: 'pws', 'PwsName' and 'geometry'.  Note 'pws' contains ids. Please note that state_name must be the full name of a state in USA.

        [ Definition 12 ]
        We have the following function to load all FRS facilities in a given sector and a state as a GeoDataFrame:
              load_FRS_facilities(state: str, naics_name: str, limit: int = 1000)
        with the following columns: 
             'facilityName', 'industryCodes', 'countyName', 'stateName', 'frsId', 'triId', 
             'rcraId', 'airId', 'npdesId', 'envInterestTypes', 'facility', 'geometry'
        Please note the argument state must be "Illinois", "Maine", or "Ohio", and the argument naics_name must be:
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
            Carpet and Upholstery Cleaning Services
            Solid Waste Landfill

        [ Definition 13 ]
        We have the following function to load all USDA ARS sites (with or without pesticide) as a GeoDataFrame:
            load_usda_ars_sites(state=None, pesticide=False)
        where state should be two letters abbreviation of a state. It loads all sites when state=None.

        [ Definition 14 ]
        We have the following function to load Military Bases in USA from an ArcGIS Feature service as a GeoDataFrame:
            load_military_bases(where: str = "countryName='usa'", bbox: Optional[List[float]] = None)
        The parameter "where" is a condition and "bbox" is a bounding box. The returned GeoDataFrame returns the following columns:
            'geometry', 'OBJECTID', 'countryName', 'featureDescription', 'featureName', 'isCui', 'isFirrmaSite', 
            'isJointBase', 'mediaId', 'mirtaLocationsIdpk', 'sdsId', 'siteName', 'siteOperationalStatus', 'siteReportingComponent', 
            'stateNameCode', 'Shape__Area', 'Shape__Length'
        Note that stateNameCode contains two letters abbreviations of states. Please always include the condition countryName='usa' to
        get military bases because we only want to study military bases in USA.
        
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
            gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
            # Ensure all columns from gdf2 are retained
            for col in gdf2.columns:
                if col not in gdf.columns:
                    gdf[col] = gdf2[col]
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
            gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
            # Ensure all columns from gdf2 are retained
            for col in gdf2.columns:
                if col not in gdf.columns:
                    gdf[col] = gdf2[col]
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
            gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
            # Ensure all columns from gdf2 are retained
            for col in gdf2.columns:
                if col not in gdf.columns:
                    gdf[col] = gdf2[col]
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
            gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
            gdf = gdf[gdf2.columns]
            gdf.title = "All the watersheds in Ross County in Ohio State"

        [ Example 7]
        Find all basins through which the Scioto River flows. This request means "find all basins which are 
        intersecting with the Scioto River".

        Find out if one of the available variables is a geodataframe containing the Scioto River.

        If none of the available variables are geodataframes containing the Scioto River, 
        then return the following code:
            raise Exception("The data for the Scioto River is missing. Please load it first.")

        If you found a variable which is a geodataframe containing the Scioto River, then return 
        the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for the Scioto River if you found one>
            gdf2 = load_basins("1 = 1", gdf1.total_bounds)
            gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
            gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
            # Ensure all columns from gdf2 are retained
            for col in gdf2.columns:
                if col not in gdf.columns:
                    gdf[col] = gdf2[col]
            gdf = gdf[gdf2.columns]
            gdf.title = "All the basins through which the Scioto River flows"

        [ Example 8 ]
        Find all the basins in Ohio State.

        Find out if one of the available variables is a geodataframe containing Ohio State.

        If none of the available variables are geodataframes containing Ohio State, 
        then return the following code:
            raise Exception("The data for the Ohio State is missing. Please load it first.")
        
        If you found a variable which is a geodataframe containing Ohio State, then return 
        the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for the Ohio State if you found one>
            gdf2 = load_basins("1 = 1", gdf1.total_bounds)
            gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="intersects")
            gdf = gdf[~gdf.geometry.apply(lambda geom: geom.touches(gdf1.unary_union))]
            # Ensure all columns from gdf2 are retained
            for col in gdf2.columns:
                if col not in gdf.columns:
                    gdf[col] = gdf2[col]
            gdf = gdf[gdf2.columns]
            gdf.title = "All the basins in Ohio State"

        [ Example 9 ]
        Find all census blocks within 5 miles distance to the power station dpq5d2851w52

        Find out if one of the available variables is a geodataframe containing the power station dpq5d2851w52.

        If none of the available variables are geodataframes containing the power station dpq5d2851w52, 
        then return the following code:
            raise Exception("The data for the power station dpq5d2851w52 is missing. Please load it first.")

        If you found a variable which is a geodataframe containing the power station dpq5d2851w52, 
        then return the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for the power station dpq5d2851w52 if you found one>
            gdf = load_nearby_census_blocks(gdf1.geometry.centroid.y.iloc[0], gdf1.geometry.centroid.x.iloc[0], 5)
            gdf.title = "All Census Blocks within 5 Miles Distance to the Power Station with ID 'dpq5d2851w52'"

        [ Example 10 ]
        Find the tracts of all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025.

        Find out if one of the available variables is a geodataframe containing all power stations at risk of 
        flooding in Ohio at 2 PM on July 1, 2025.

        If none of the available variables are geodataframes containing all power stations at risk of flooding 
        in Ohio at 2 PM on July 1, 2025, then return the following code:
            raise Exception("The data for all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025 is missing. Please load it first.")        

        If you found a variable which is a geodataframe containing all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025, 
        then return the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025 if you found one>
            # Get unique tract GEOIDs
            unique_geoids = gdf1["GEOID"].dropna().unique()
            all_tracts = []
            for geoid in unique_geoids:
                sample_point = gdf1[gdf1["GEOID"] == geoid].iloc[0].geometry
                lat, lon = sample_point.y, sample_point.x
                tract_gdf = load_census_tract(lat, lon)
                all_tracts.append(tract_gdf)
            gdf = gpd.GeoDataFrame(pd.concat(all_tracts, ignore_index=True))
            gdf = gdf.drop_duplicates(subset="GEOID")  # or 'GEOID' if available
            gdf.crs = "EPSG:4326"
            gdf.title = "the tracts of all power stations at risk of flooding in Ohio at 2 PM on July 1, 2025"

        [ Example 11 ]
        Find all power stations at risk of flooding in Ohio from 2 AM July 17, 2025 to 10 PM on July 18, 2025.

        You can return the following code:
            start = pd.to_datetime("2025-07-17 02:00")
            end = pd.to_datetime("2025-07-18 22:00")
            hours = pd.date_range(start, end, freq="H")
            hour_strs = hours.strftime("%Y%m%d%H")

            all_gdfs = []
            for hour in hour_strs:
                gdf_hour = load_flooded_power_stations(hour, scope="39")  
                gdf_hour["Date"] = hour
                all_gdfs.append(gdf_hour)
            gdf = gpd.GeoDataFrame(pd.concat(all_gdfs, ignore_index=True), crs="EPSG:4326")
            gdf.title = "All power stations at risk of flooding in Ohio from 2 AM July 17, 2025 to 10 PM on July 18, 2025"

        [ Example 12 ]
        Find all public water systems in Ross county, Ohio.

        Find out if one of the available variables is a geodataframe containing Ross county, Ohio.

        If none of the available variables are geodataframes containing Ross county, Ohio:
            raise Exception("The data for Ross county, Ohio is missing. Please load it first.")    

        If you found a variable which is a geodataframe containing Ross county, Ohio, 
        then return the valid Python code in the following format:

            gdf1 = <replace by the variable of the geodataframe for Ross county, Ohio if you found one>
            gdf2 = load_public_water_systems(state_name="ohio", limit=3000)
            gdf1 = gdf1.set_crs("EPSG:4326")   # if it's in lon/lat
            gdf2 = gdf2.set_crs("EPSG:4326") 
            gdf = gpd.sjoin(gdf2, gdf1, how="inner", predicate="within")
            gdf.title = "All public water systems in Ross county, Ohio"

        [ Example 13 ]
        Find PFSA contamination observations within 100 meters to Presumpscot River.

        Find out if one of the available variables is a geodataframe containing Presumpscot River.

        If none of the available variables are geodataframes containing Presumpscot River:
            raise Exception("The data for Presumpscot River is missing. Please load it first.")    

        If you found a variable which is a geodataframe containing Presumpscot River, 
        then return the valid Python code in the following format:
            gdf1 = <replace by the variable of the geodataframe for Presumpscot River if you found one>
            gdf2 = load_PFAS_contamiation_observations()
            gdf1 = gdf1.set_crs("EPSG:4326")  
            gdf2 = gdf2.set_crs("EPSG:4326") 
            gdf1 = gdf1.to_crs(gdf1.estimate_utm_crs())  # Presumpscot River
            gdf2 = gdf2.to_crs(gdf1.crs)                 
            joined = gpd.sjoin_nearest(
                        gdf2, gdf1,
                        max_distance=10,        # 10 meters
                        distance_col="distance_to_river"
            )
            cols_to_keep = gdf2_utm.columns.to_list() + ["distance_to_river"]
            gdf = joined[cols_to_keep].copy()
            gdf = gdf.to_crs("EPSG:4326")
            gdf.title = "PFAS contamination observations within 10 meters of Presumpscot River"

       The request "Identify FRS solid waste landfill facilities within 1 km of Androscoggin river" should use the similar way to calcuate.

       [ Example 14 ]
       Find all census tracts located downstream of the Presumpscot River from PFAS contamination observations within 100 meters of the river.

       Find out if one of available variable is a geodataframe containing the Presumpscot River and another of the available variables is a 
       geodataframe containing PFAS contamination observations within 100 meters of the Presumpscot River.

       If none of the available variables are geodataframes containing Presumpscot River:
            raise Exception("The data for Presumpscot River is missing. Please load it first.")   

       If none of the available variables are geodataframes containing PFAS contamination observations within 100 meters of the Presumpscot River:
            raise Exception("The data for PFAS contamination observations within 100 meters of the Presumpscot River is missing. Please load it first.")   

       If you found a variable which is a geodataframe containing Presumpscot River and another variable which is a geodataframe containing PFAS contamination observations within 100 meters of the Presumpscot River, 
       then return the valid Python code in the following format:
            gdf_river = <replace by the variable of the geodataframe for the Presumpscot River if you found one>
            gdf_pfas = <replace by the variable of the geodataframe for PFAS contamination observations within 100 meters of the Presumpscot River if you found one>
            gdf = downstream_tracts(gdf_river, gdf_pfas, flow_dir='south')
            gdf = gdf.set_crs("EPSG:4326")
            gdf.title = "all census tracts located downstream of the Presumpscot River from PFAS contamination observations within 100 meters of the river"

       [ Example 15 ]
       Find all public water systems in Maine containing PFAS contamination observations.

       You can return the following code:
            gdf_pws = load_public_water_systems(state_name="maine", limit=2000)
            gdf_pfas = load_PFAS_contamiation_observations()
            gdf_pws = gdf_pws.set_crs("EPSG:4326")
            gdf_pfas = gdf_pfas.set_crs("EPSG:4326")
            gdf = gpd.sjoin(gdf_pws, gdf_pfas, how="inner", predicate="intersects")
            gdf = gdf[gdf_pws.columns]
            gdf = gdf.drop_duplicates(subset='geometry')  
            gdf.title = "All public water systems in Maine containing PFAS contamination observations"

        Please ensure that the returned gdf does not contain duplicate rows by using gdf = gdf.drop_duplicates(subset='geometry').

        [ Example 16 ]
        Find all PFSA contamination observations within public water systems in Maine .

        You can return the following code:
            gdf_pfas = load_PFAS_contamiation_observations()
            gdf_pws = load_public_water_systems(state_name="maine", limit=2000)
            gdf_pfas = gdf_pfas.set_crs("EPSG:4326")
            gdf_pws = gdf_pws.set_crs("EPSG:4326")
            gdf = gpd.sjoin(gdf_pfas, gdf_pws, how="inner", predicate="intersects")
            gdf = gdf[gdf_pfas.columns]
            gdf = gdf.drop_duplicates(subset='geometry')  # or 'pfas_id' if available
            gdf.title = "All PFSA contamination observations within public water systems in Maine"

        Please ensure that the returned gdf does not contain duplicate rows by using gdf = gdf.drop_duplicates(subset='geometry').
        
        [ Example 17 ]
        Find all PFSA contamination observations within 800 meters from FRS water supply and irrigation facilities in Maine.

        You can return the following code:
            gdf_pfas = load_PFAS_contamiation_observations()
            gdf_frs = load_FRS_facilities(state="Maine", naics_name="Water Supply and Irrigation", limit=1000)
            gdf_pfas = gdf_pfas.set_crs("EPSG:4326")
            gdf_frs = gdf_frs.set_crs("EPSG:4326")
            gdf_frs_utm = gdf_frs.to_crs(gdf_frs.estimate_utm_crs())
            gdf_pfas_utm = gdf_pfas.to_crs(gdf_frs_utm.crs)
            max_distance_meters = 800
            joined = gpd.sjoin_nearest(
                gdf_pfas_utm, gdf_frs_utm,
                max_distance=max_distance_meters,
                distance_col="distance_to_frs"
            )
            gdf = joined[joined["distance_to_frs"] <= max_distance_meters].copy()
            gdf = gdf[gdf_pfas.columns]
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.drop_duplicates(subset='geometry')  
            gdf.title = "All PFSA contamination observations within 800 meters from FRS water supply and irrigation facilities in Maine"

        Please ensure that the returned gdf does not contain duplicate rows by using gdf = gdf.drop_duplicates(subset='geometry').

        [ Example 18 ]
        Find all FRS water supply and irrigation facilities in Maine within 800 meters from PFSA contamination observations.

        You can return the following code:
            gdf_frs = load_FRS_facilities(state="Maine", naics_name="Water Supply and Irrigation", limit=1000)
            gdf_pfas = load_PFAS_contamiation_observations()
            gdf_frs = gdf_frs.set_crs("EPSG:4326")
            gdf_pfas = gdf_pfas.set_crs("EPSG:4326")
            gdf_frs_utm = gdf_frs.to_crs(gdf_frs.estimate_utm_crs())
            gdf_pfas_utm = gdf_pfas.to_crs(gdf_frs_utm.crs)
            max_distance_meters = 800
            joined = gpd.sjoin_nearest(
                gdf_frs_utm, gdf_pfas_utm, 
                max_distance=max_distance_meters,
                distance_col="distance_to_pfas"
            )
            gdf = joined[joined["distance_to_pfas"] <= max_distance_meters].copy()
            gdf = gdf[gdf_frs.columns]
            gdf = gdf.to_crs("EPSG:4326")
            gdf = gdf.drop_duplicates(subset='geometry')  
            gdf.title = "All FRS Water Supply and Irrigation Facilities in Maine within 800 meters from PFAS Contamination Observations"

        Please ensure that the returned gdf does not contain duplicate rows by using gdf = gdf.drop_duplicates(subset='geometry').

        [ Example 19 ]
        Find all census tracts located downstream of the Androscoggin River from FRS solid waste landfill facilities within 1 km of the river.

        Find out if one of available variable is a geodataframe containing the Androscoggin River and another of the available variables is a 
        geodataframe containing FRS solid waste landfill facilities within 1 km of the Androscoggin River.

       If none of the available variables are geodataframes containing Androscoggin River:
            raise Exception("The data for Androscoggin River is missing. Please load it first.")   

       If none of the available variables are geodataframes containing FRS solid waste landfill facilities within 1 km of the Androscoggin River:
            raise Exception("The data for FRS solid waste landfill facilities within 1 km of the Androscoggin River is missing. Please load it first.")   

       If you found a variable which is a geodataframe containing Androscoggin River and another variable which is a geodataframe containing FRS solid waste landfill facilities within 1 km of the Androscoggin River, 
       then return the valid Python code in the following format:
            gdf_river = <replace by the variable of the geodataframe for the Androscoggin River if you found one>
            gdf_frs = <replace by the variable of the geodataframe for FRS solid waste landfill facilities within 1 km of the Androscoggin River if you found one>
            gdf = downstream_tracts(gdf_river, gdf_frs, flow_dir='south')
            gdf = gdf.set_crs("EPSG:4326")
            gdf.title = "all census tracts located downstream of the Androscoggin River from FRS solid waste landfill facilities within 1 km of the river"

        [ Example 20 ]
        Identify all buildings that were at risk of flooding in Ohio at 2:00 PM on August 1, 2025 within 300 meters from FRS Sewage Treatment facilities.

        You can return the following code:
            gdf_frs = load_FRS_facilities(state="Ohio", naics_name="Sewage Treatment", limit=1000)
            gdf_buildings = load_blooded_buildings("2025080114", "39")
            # then code for calculating distance within 300 meters    

        [ Example 21 ]
        Identify all buildings that were at risk of flooding in Ohio at 2:00 PM on August 1, 2025 within 300 meters from FRS Sewage Treatment facilities

        You can return the following code:
            gdf_frs = load_FRS_facilities(state="Ohio", naics_name="Sewage Treatment", limit=1000)
            gdf_buildings = load_blooded_buildings("2025080114", "39")
            # then code for calculating distance within 300 meters    

        [ Example 22 ]
        Find military bases in Maine containing PFAS contamination observations.

        You can return the following code:
            gdf_military = load_military_bases("countryName='usa' AND stateNameCode='ME'")
            gdf_pfas = load_PFAS_contamiation_observations()
            gdf_military = gdf_military.set_crs("EPSG:4326")
            gdf_pfas = gdf_pfas.set_crs("EPSG:4326")
            gdf = gpd.sjoin(gdf_military, gdf_pfas, how="inner", predicate="intersects")
            gdf = gdf[gdf_military.columns]
            gdf = gdf.drop_duplicates(subset='geometry')  
            gdf.title = "Military bases in Maine containing PFAS contamination observations"

        [ Example 21 ]
        Find PFAS contamination observations in military bases in Maine.

        You can return the following code:
            gdf_pfas = load_PFAS_contamiation_observations()
            gdf_military = load_military_bases("countryName='usa' AND stateNameCode='ME'")
            gdf_pfas = gdf_pfas.set_crs("EPSG:4326")
            gdf_military = gdf_military.set_crs("EPSG:4326")
            gdf = gpd.sjoin(gdf_pfas, gdf_military, how="inner", predicate="within")
            gdf = gdf[gdf_pfas.columns]
            gdf = gdf.drop_duplicates(subset='geometry')  
            gdf.title = "PFAS contamination observations in military bases in Maine"
            
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

class ArcGISFeatureLoader:
    def __init__(self, url: str, batch_size: int = 100, max_workers: int = 4, max_retries: int = 3):
        """
        Initialize the ArcGIS Feature Service loader.
        
        Args:
            url: The base URL of the ArcGIS Feature Service
            batch_size: Number of records to fetch per request
            max_workers: Maximum number of concurrent workers
            max_retries: Maximum number of retry attempts per failed request
        """
        self.url = url
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.max_retries = max_retries

    def get_total_record_count(self, where: str) -> int:
        """Fetch the total number of records available."""
        params = {
            "where": where,
            "returnCountOnly": "true",
            "f": "json"
        }
        response = requests.get(self.url + "/query", params=params)
        response.raise_for_status()
        return response.json().get("count", 0)

    def fetch_batch(self, where: str, offset: int, bbox: Optional[List[float]] = None) -> List[Dict[str, Any]]:
        """
        Fetch a batch of features with retry logic.
        
        Args:
            where: SQL where clause
            offset: Starting record offset
            bbox: Optional bounding box [minx, miny, maxx, maxy]
        """
        params = {
            "where": where,
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326",
            "resultOffset": offset,
            "resultRecordCount": self.batch_size
        }
        
        if bbox:
            params.update({
                "geometry": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",
                "geometryType": "esriGeometryEnvelope",
                "spatialRel": "esriSpatialRelIntersects"
            })

        for attempt in range(self.max_retries):
            try:
                response = requests.get(self.url + "/query", params=params)
                response.raise_for_status()
                data = response.json()
                return data.get('features', [])
            except Exception as e:
                if attempt == self.max_retries - 1:
                    logger.error(f"Failed to fetch batch at offset {offset} after {self.max_retries} attempts: {str(e)}")
                    raise
                time.sleep(1 * (attempt + 1))  # Exponential backoff

    def load_features(self, where: str = "1=1", bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
        """
        Load all features from the service using concurrent requests.
        
        Args:
            where: SQL where clause
            bbox: Optional bounding box [minx, miny, maxx, maxy]
            
        Returns:
            GeoDataFrame containing all features
        """
        total_records = self.get_total_record_count(where)
        logger.info(f"Total records to fetch: {total_records}")
        
        if total_records == 0:
            return gpd.GeoDataFrame(columns=['geometry'])

        offsets = range(0, total_records, self.batch_size)
        all_features = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_offset = {
                executor.submit(self.fetch_batch, where, offset, bbox): offset 
                for offset in offsets
            }
            
            completed = 0
            for future in concurrent.futures.as_completed(future_to_offset):
                offset = future_to_offset[future]
                try:
                    features = future.result()
                    all_features.extend(features)
                    completed += len(features)
                    logger.info(f"Progress: {completed}/{total_records} features ({(completed/total_records)*100:.1f}%)")
                except Exception as e:
                    logger.error(f"Failed to fetch batch at offset {offset}: {str(e)}")

        if not all_features:
            return gpd.GeoDataFrame(columns=['geometry'])

        gdf = gpd.GeoDataFrame.from_features(all_features)
        logger.info(f"Successfully loaded {len(gdf)} features")
        return gdf


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

def load_basins(where, bbox):
    self_url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Watershed_Boundary_Dataset_HUC_6s/FeatureServer/0"
    wkid = "3857"
    return get_arcgis_features(self_url, where, bbox)  

def load_basins_2(where: str = "1=1", bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
    """Load watershed boundary dataset using concurrent fetching."""
    url = "https://services.arcgis.com/P3ePLMYs2RVChkJx/ArcGIS/rest/services/Watershed_Boundary_Dataset_HUC_6s/FeatureServer/0"
    
    loader = ArcGISFeatureLoader(
        url=url,
        batch_size=100,
        max_retries=3
    )   
    if bbox is None and where == "1 = 1":
        raise Exception("Your request returned a large number of basins. Please refine your search.")
        # bbox = [ -89.5, 36.5, -80.5, 42.0 ]
    gdf = loader.load_features(where=where, bbox=bbox)
    return gdf

def load_census_block(latitude, longitude):
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query"
    params = {
        "f": "geojson",
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true"
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    gdf = gpd.read_file(resp.text)
    if gdf.empty:
        raise ValueError("No census block found at the given location.")
    return gdf

def load_nearby_census_blocks(lat, lon, radius_miles=5):
    # Convert miles to meters
    radius_m = radius_miles * 1609.34
    
    # Project WGS84 to an equal-area projection for buffering
    wgs84 = pyproj.CRS("EPSG:4326")
    aeqd = pyproj.CRS(proj="aeqd", lat_0=lat, lon_0=lon, datum="WGS84")
    project_to_aeqd = pyproj.Transformer.from_crs(wgs84, aeqd, always_xy=True).transform
    project_to_wgs84 = pyproj.Transformer.from_crs(aeqd, wgs84, always_xy=True).transform
    
    # Create buffer in meters around point and project back to WGS84
    point = Point(lon, lat)
    buffer = transform(project_to_aeqd, point).buffer(radius_m)
    buffer_wgs84 = transform(project_to_wgs84, buffer)

    # Convert geometry to ESRI JSON
    buffer_geojson = gpd.GeoSeries([buffer_wgs84]).__geo_interface__['features'][0]['geometry']

    esri_geometry = {
        "rings": buffer_geojson['coordinates'],
        "spatialReference": {"wkid": 4326}
    }

    # Query the FeatureServer
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/2/query"
    params = {
        "f": "geojson",
        "geometry": json.dumps(esri_geometry),
        "geometryType": "esriGeometryPolygon",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "inSR": "4326",
        "outSR": "4326",
        "returnGeometry": "true"
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    
    return gpd.read_file(io.StringIO(resp.text))

def load_census_tract(latitude, longitude):
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query"
    params = {
        "f": "geojson",
        "geometry": f"{longitude},{latitude}",
        "geometryType": "esriGeometryPoint",
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true"
    }

    resp = requests.get(url, params=params)
    resp.raise_for_status()
    gdf = gpd.read_file(resp.text)
    if gdf.empty:
        raise ValueError("No census tract found at the given location.")
    return gdf

def get_tracts_for_geometry(geometry, retries=3, buffer_distance=0.0001):
    url = "https://tigerweb.geo.census.gov/arcgis/rest/services/TIGERweb/Tracts_Blocks/MapServer/0/query"
    
    # Handle buffering for lines to address precision/gap issues
    if buffer_distance > 0 and isinstance(geometry, (LineString, MultiLineString)):
        geometry = geometry.buffer(buffer_distance)
    
    if isinstance(geometry, Point):
        geometry_type = "esriGeometryPoint"
        geometry_param = {"x": geometry.x, "y": geometry.y}
        if not (-125 <= geometry.x <= -66 and 24 <= geometry.y <= 49):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    elif isinstance(geometry, (LineString, MultiLineString)):
        geometry_type = "esriGeometryPolyline"
        geometry = geometry.simplify(tolerance=0.0001, preserve_topology=True)
        if isinstance(geometry, LineString):
            coords = list(geometry.coords)
            paths = [coords]
        else:
            coords = [coord for line in geometry.geoms for coord in line.coords]
            paths = [list(line.coords) for line in geometry.geoms]
        if not all(-125 <= lon <= -66 and 24 <= lat <= 49 and np.isfinite(lon) and np.isfinite(lat) for lon, lat in coords):
            return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
        geometry_param = {"paths": paths}
    elif isinstance(geometry, (shapely.geometry.Polygon, shapely.geometry.MultiPolygon)):
        geometry_type = "esriGeometryPolygon"
        def get_rings(geom):
            if isinstance(geom, shapely.geometry.Polygon):
                return [list(map(list, geom.exterior.coords))]
            else:
                rings = []
                for p in geom.geoms:
                    rings.append(list(map(list, p.exterior.coords)))
                return rings
        geometry_param = {"rings": get_rings(geometry)}
    else:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    params = {
        "f": "geojson",
        "geometry": json.dumps(geometry_param),
        "geometryType": geometry_type,
        "inSR": 4326,
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "GEOID,STATE,COUNTY,TRACT,NAME",
        "returnGeometry": "true"
    }
    
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            gdf = gpd.read_file(resp.text)
            if gdf.empty:
                return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
            gdf.crs = "EPSG:4326"
            return gdf[['GEOID', 'STATE', 'COUNTY', 'TRACT', 'NAME', 'geometry']]
        except:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")

def get_tracts_for_river(river_gdf, max_segments=None):
    if river_gdf.crs != "EPSG:4326":
        river_gdf = river_gdf.to_crs("EPSG:4326")
    
    tract_list = []
    
    for _, row in river_gdf.iterrows():
        geometry = row.geometry
        if geometry.is_empty:
            continue
        if isinstance(geometry, MultiLineString):
            lines = list(geometry.geoms)
            if max_segments is not None:
                lines = lines[:max_segments]
            for sub_geom in lines:
                tracts = get_tracts_for_geometry(sub_geom)
                if not tracts.empty:
                    tract_list.append(tracts)
                time.sleep(0.2)
        else:
            tracts = get_tracts_for_geometry(geometry)
            if not tracts.empty:
                tract_list.append(tracts)
            time.sleep(0.2)
    
    if not tract_list:
        centroid = river_gdf.geometry.unary_union.centroid
        tracts = load_census_tract(centroid.y, centroid.x)
        if not tracts.empty:
            tract_list.append(tracts)
    
    if not tract_list:
        return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")
    
    combined = gpd.GeoDataFrame(pd.concat(tract_list, ignore_index=True), crs="EPSG:4326")
    return combined.drop_duplicates(subset='GEOID')

def downstream_tracts(river_gdf, point_gdf, flow_dir='south'):
    if river_gdf.crs is None:
        river_gdf = river_gdf.set_crs("EPSG:4326")
    else:
        river_gdf = river_gdf.to_crs("EPSG:4326")
    
    if point_gdf.crs is None:
        point_gdf = point_gdf.set_crs("EPSG:4326")
    else:
        point_gdf = point_gdf.to_crs("EPSG:4326")
    
    results = []
    
    for idx, row in point_gdf.iterrows():
        pt = row.geometry
        pt_lat = pt.y
        pt_lon = pt.x
        
        # Combine all river geometries into a single LineString or MultiLineString
        river_geom = river_gdf.geometry.unary_union
        if isinstance(river_geom, MultiLineString):
            # Merge MultiLineString into a single LineString if possible
            river_geom = shapely.ops.linemerge(river_geom)
        
        if not isinstance(river_geom, LineString):
            continue  # Skip if we can't get a single LineString
        
        # Project the point onto the river to find the closest point
        projected_distance = river_geom.project(pt)
        river_length = river_geom.length
        
        # Extract the downstream portion (from projected point to end)
        # Assuming river is digitized upstream to downstream
        downstream_geom = shapely.ops.substring(river_geom, projected_distance, river_length)
        
        if downstream_geom.is_empty or not downstream_geom.is_valid:
            continue
        
        # Create a GeoDataFrame with the downstream geometry
        river_downstream = gpd.GeoDataFrame(geometry=[downstream_geom], crs="EPSG:4326")
        
        # Get tracts for the downstream river segment
        tracts = get_tracts_for_river(river_downstream)
        if not tracts.empty:
            tracts = tracts.copy()
            tracts['source_point'] = idx
            tracts['point_lat'] = pt_lat
            tracts['point_lon'] = pt_lon
            results.append(tracts)
    
    if results:
        combined = gpd.GeoDataFrame(pd.concat(results, ignore_index=True), crs="EPSG:4326")
        return combined.drop_duplicates(subset='GEOID')
    return gpd.GeoDataFrame(geometry=[], crs="EPSG:4326")


def fetch_flood_impacts(
    date: str,
    fips: str = "county",
    feature_type: str = "power",
    scope: Optional[Union[str, List[str]]] = None,
    base_url: str = "https://staging.api-flooding.data2action.tech/v0/impacts/structures",
    max_retries: int = 3,
    delay_between_requests: float = 0.1
) -> gpd.GeoDataFrame:
    """
    Fetch flood impact data from the API and return as a GeoDataFrame.
    
    Parameters:
    -----------
    date : str
        Date in format YYYYMMDDHH (e.g., "2025071414")
    fips : str, optional
        FIPS level - one of "state", "county", "tract", "block-group" (default: "county")
    feature_type : str, optional
        Type of feature - one of "building", "ust", "power" (default: "power")
    scope : str or list of str, optional
        Geographic scope filter. Can be:
        - State FIPS code (e.g., "39" for Ohio)
        - County FIPS code (e.g., "39009" for Athens County, Ohio)
        - List of FIPS codes
        If None, defaults to Ohio ("39") and Kentucky ("21")
    base_url : str, optional
        Base URL for the API
    max_retries : int, optional
        Maximum number of retries for failed requests (default: 3)
    delay_between_requests : float, optional
        Delay between API requests in seconds (default: 0.1)
    
    Returns:
    --------
    geopandas.GeoDataFrame
        GeoDataFrame containing all flood impact structures
    
    Raises:
    -------
    ValueError
        If invalid parameters are provided
    requests.RequestException
        If API requests fail after max retries
    """
    
    # Validate parameters
    valid_fips = ["state", "county", "tract", "block-group"]
    valid_feature_types = ["building", "ust", "power"]
    
    if fips not in valid_fips:
        raise ValueError(f"fips must be one of {valid_fips}")
    
    if feature_type not in valid_feature_types:
        raise ValueError(f"feature_type must be one of {valid_feature_types}")
    
    # Default to Ohio (39) and Kentucky (21) if no scope specified
    if scope is None:
        scope = ["39", "21"]  # Ohio and Kentucky FIPS codes
    
    # Ensure scope is a list
    if isinstance(scope, str):
        scope = [scope]
    
    # Validate date format
    if len(date) != 10 or not date.isdigit():
        raise ValueError("date must be in format YYYYMMDDHH (e.g., '2025071414')")
    
    all_features = []
    
    print(f"Fetching {feature_type} data for {fips} level on {date} in scope {scope}...")
    
    # Process each scope separately since API might handle multiple scopes differently
    for scope_item in scope:
        page = 0
        
        print(f"Processing scope: {scope_item}")
        
        while True:
            # Construct parameters
            params = {
                "date": date,
                "fips": fips,
                "feature-type": feature_type,
                "scope": scope_item,
                "response-format": "geojson",
                "page": page,
                "size": 1000  # Maximum page size
            }

            # Define headers with API key
            headers = {
                "x-api-key": "maj6OM1L77141VXiH7GMy1iLRWmFI88M5JVLMHn7"
            }

            # Make request with retry logic
            response = None
            for attempt in range(max_retries):
                try:
                    response = requests.get(base_url, params=params, headers=headers, timeout=30)
                    if response.status_code == 404:
                        print(f"No data available for {date} (404 Not Found). Skipping.")
                        return gpd.GeoDataFrame(
                            columns=["fips", "feature-type", "geometry"],
                            geometry="geometry",
                            crs="EPSG:4326"
                        )
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt == max_retries - 1:
                        raise requests.RequestException(f"Failed to fetch data after {max_retries} attempts: {e}")
                    print(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(delay_between_requests * (2 ** attempt))  # Exponential backoff
            
            try:
                data = response.json()
            except ValueError as e:
                raise ValueError(f"Invalid JSON response: {e}")
            
            # Extract features from response
            if "structures" not in data:
                raise ValueError("Unexpected response format: missing 'structures' key")
            
            structures = data["structures"]
            features = structures.get("features", [])
            
            if not features:
                break
            
            all_features.extend(features)
            
            # Check if we have more pages
            properties = structures.get("properties", {})
            index_info = properties.get("index", {})
            total = properties.get("total", 0)
            end_index = index_info.get("end", 0)
            
            print(f"Scope {scope_item}, Page {page}: Retrieved {len(features)} features (total so far: {len(all_features)})")
            
            if end_index >= total or len(features) < 1000:
                break
            
            page += 1
            
            # Small delay to be respectful to the API
            time.sleep(delay_between_requests)
    
    print(f"Completed: Retrieved {len(all_features)} total features")
    
    # Convert to GeoDataFrame
    if not all_features:
        # Return empty GeoDataFrame with correct schema
        return gpd.GeoDataFrame(
            columns=["fips", "feature-type", "geometry"],
            geometry="geometry",
            crs="EPSG:4326"
        )
    
    # Extract data for DataFrame
    rows = []
    for feature in all_features:
        props = feature["properties"]
        coords = feature["geometry"]["coordinates"]
        
        row = {
            "fips": props["fips"],
            "feature-type": props["feature-type"],
            "geometry": Point(coords[0], coords[1])  # lon, lat
        }
        rows.append(row)
    
    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(rows, crs="EPSG:4326")
    gdf["Date"] = date
    return gdf

def load_flooded_power_stations(date: str, scope) -> gpd.GeoDataFrame:
    return fetch_flood_impacts(date, fips="tract", feature_type="power", scope=scope) 

def load_flooded_buildings(date: str, scope) -> gpd.GeoDataFrame:
    return fetch_flood_impacts(date, fips="tract", feature_type="building", scope=scope) 
    
def load_PFAS_contamiation_observations() -> gpd.GeoDataFrame:
    """
    Fetch PFAS contaminant samples exceeding thresholds and return as GeoDataFrame.
    """
#     endpoint_url = "https://frink.apps.renci.org/qlever-geo/sparql"

#     query = """
# PREFIX pfas: <http://sawgraph.spatialai.org/v1/pfas#>
# PREFIX coso: <http://sawgraph.spatialai.org/v1/contaminoso#>
# PREFIX sosa: <http://www.w3.org/ns/sosa/>
# PREFIX geo: <http://www.opengis.net/ont/geosparql#>
# PREFIX qudt: <http://qudt.org/schema/qudt/>
# PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

# SELECT DISTINCT
#   ?wkt
#   (SAMPLE(?obs) AS ?Obs)
#   (SAMPLE(?substance) AS ?Substance)
#   (MAX(?date) AS ?Date)
#   (SAMPLE(?value) AS ?Value)
#   (SAMPLE(?unit) AS ?Unit)
#   (SAMPLE(?samplePoint) AS ?SamplePoint)
# WHERE {
#   ?obs a pfas:PFAS-ContaminantObservation ;
#        a coso:SampleContaminantObservation ;
#        sosa:hasResult ?result ;
#        coso:observedAtSamplePoint ?samplePoint ;
#        coso:ofSubstance ?substance ;
#        coso:sampledTime ?date .

#   ?result qudt:quantityValue ?qv .
#   ?qv qudt:numericValue ?value ;
#       qudt:unit ?unit .

#   ?samplePoint geo:hasGeometry/geo:asWKT ?wkt .

#   FILTER(BOUND(?wkt) && STRLEN(STR(?wkt)) > 0) .
#   FILTER(?date >= "2000-01-01"^^xsd:date) .

#   VALUES (?substanceVal ?limitVal ?unitVal) {
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOS_A> 10 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOA_A> 10 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.SUM_PFOA_PFOS> 20 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFNA_A> 5 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFDA_A> 5 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFBS_A> 5 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOSA> 1 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFTEA_A> 1 <http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOS> 70 <http://qudt.org/vocab/unitNanoGM-PER-L>)
#     (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOA_A> 70 <http://qudt.org/vocab/unitNanoGM-PER-L>)
#   }

#   FILTER(?substance = ?substanceVal && ?value > ?limitVal && ?unit = ?unitVal)
# }
# GROUP BY ?wkt
# LIMIT 1000
# """

    endpoint_url = "https://frink.apps.renci.org/federation/sparql"
    query = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX coso: <http://w3id.org/coso/v1/contaminoso#>
PREFIX qudt: <http://qudt.org/schema/qudt#>

SELECT ?wkt
       (SAMPLE(?observation) AS ?Obs)
       (SAMPLE(?substance) AS ?Substance)
       (MAX(?obsDate) AS ?Date)
       (SAMPLE(?result_value) AS ?Value)
       (SAMPLE(?unit) AS ?Unit)
       (SAMPLE(?samplePoint) AS ?SamplePoint)
WHERE {
  ?observation rdf:type coso:ContaminantObservation ;
               coso:observedAtSamplePoint ?samplePoint ;
               coso:ofSubstance ?substance ;
               coso:hasResult ?result ;
               coso:observedTime ?obsDate .

  ?samplePoint rdf:type coso:SamplePoint ;
               geo:hasGeometry/geo:asWKT ?wkt .

  ?result coso:measurementValue ?result_value ;
          coso:measurementUnit ?unit .

   VALUES (?substanceVal ?limitVal ?unitVal) {
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOS_A> 4 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOA_A> 4 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFNA_A> 10 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFHXA_A> 10 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFBS_A> 10 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFTEA_A> 10 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOS> 4 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
    (<http://sawgraph.spatialai.org/v1/me-egad#parameter.PFOA> 4 <http://qudt.org/vocab/unit/NanoGM-PER-L>)
  }

  FILTER(?substance = ?substanceVal && ?unit = ?unitVal && ?result_value > ?limitVal)
}
GROUP BY ?wkt
"""
    
    df = sparql_dataframe.get(endpoint_url, query)

    # Convert WKT to geometry
    if "wkt" in df.columns:
        df = df.dropna(subset=["wkt"]).copy()
        df["geometry"] = df["wkt"].apply(wkt.loads)
        df = df.drop(columns=["wkt"])
    else:
        df["geometry"] = None

    # Optionally add medium
    def get_medium(unit):
        if unit == "http://sawgraph.spatialai.org/v1/me-egad#unit.NG-G":
            return "soil/tissue"
        elif unit == "http://qudt.org/vocab/unitNanoGM-PER-L":
            return "water"
        else:
            return "unknown"

    df["medium"] = df["Unit"].apply(get_medium)
    
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    return gdf    

def load_public_water_systems(state_name: str = "maine", limit: int = 3000) -> gpd.GeoDataFrame:
#     endpoint_url = "https://frink.apps.renci.org/qlever-geo/sparql"
    
#     query = f"""
# PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
# PREFIX schema: <https://schema.org/>
# PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
# PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
# PREFIX geo: <http://www.opengis.net/ont/geosparql#>
# PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
# PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
# PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>

# SELECT ?pws (SAMPLE(?pwsName) AS ?PwsName) (SAMPLE(?pwsGeometry) AS ?anyPwsGeometry)
# FROM <https://frink.renci.org/kg/geoconnex>
# FROM <https://frink.renci.org/kg/s2/13-13>
# FROM <https://frink.renci.org/kg/spatialkg>
# WHERE {{
#     ?pws schema:name ?pwsName ;
#          geo:hasGeometry/geo:asWKT ?pwsGeometry.
#     FILTER(STRSTARTS(STR(?pws), "https://geoconnex.us/ref/pws/"))

#     ?state rdf:type kwg-ont:AdministrativeRegion_1 ;
#            geo:hasGeometry/geo:asWKT ?stateGeom ;
#            rdfs:label ?stateLabel .
#     FILTER(CONTAINS(LCASE(?stateLabel), "{state_name.lower()}"))

#     FILTER (geof:sfIntersects(?pwsGeometry, ?stateGeom))
# }}
# GROUP BY ?pws
# LIMIT {limit}
# """

    endpoint_url = "https://frink.apps.renci.org/federation/sparql"
    
    query = f"""
PREFIX schema: <https://schema.org/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX sockg: <https://idir.uta.edu/sockg-ontology/docs/>

SELECT DISTINCT ?pws (SAMPLE(?pwsName) AS ?PwsName) (SAMPLE(?pwsGeometry) AS ?anyPwsGeometry)
WHERE {{
    ?pws schema:name ?pwsName ;
         geo:hasGeometry/geo:asWKT ?pwsGeometry.
    FILTER(STRSTARTS(STR(?pws), "https://geoconnex.us/ref/pws/")) .
  
    ?state rdf:type kwg-ont:AdministrativeRegion_1 ;
           geo:hasGeometry/geo:asWKT ?stateGeom ;
           rdfs:label ?stateLabel .
    FILTER(CONTAINS(LCASE(?stateLabel), "{state_name.lower()}")) 
    FILTER (geof:sfIntersects(?pwsGeometry, ?stateGeom)) 
}}
GROUP BY ?pws
LIMIT {limit}
"""
    
    # Fetch data
    df = sparql_dataframe.get(endpoint_url, query)

    # Convert WKT to geometry
    df = df.dropna(subset=["anyPwsGeometry"]).copy()
    df["geometry"] = df["anyPwsGeometry"].apply(wkt.loads)
    df = df.drop(columns=["anyPwsGeometry"])

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf

# Allowed states and NAICS industries
ALLOWED_STATES = ["Illinois", "Maine", "Ohio"]
ALLOWED_NAICS = [
    "Waste Treatment and Disposal",
    "Converted Paper Manufacturing",
    "Water Supply and Irrigation",
    "Sewage Treatment",
    "Plastics Product Manufacturing",
    "Textile and Fabric Finishing and Coating",
    "Basic Chemical Manufacturing",
    "Paint, Coating, and Adhesive Manufacturing",
    "Aerospace Product and Parts",
    "Drycleaning and Laundry Services",
    "Carpet and Upholstery Cleaning Services",
    "Solid Waste Landfill",
]

def load_FRS_facilities(state: str, naics_name: str, limit: int = 1000) -> gpd.GeoDataFrame:
    """
    Load facilities from the FRS dataset for a given state and NAICS industry name.

    Parameters:
        state (str): State name, e.g., "Illinois", "Maine", "Ohio".
        naics_name (str): NAICS industry name, e.g., "Waste Treatment and Disposal".
        limit (int): Maximum number of facilities to fetch (default 1000).

    Returns:
        gpd.GeoDataFrame: Facilities with geometry and other attributes.
    """

    # Validate parameters
    if state not in ALLOWED_STATES:
        raise ValueError(f"Invalid state '{state}'. Allowed states: {ALLOWED_STATES}")
    if naics_name not in ALLOWED_NAICS:
        raise ValueError(f"Invalid NAICS name '{naics_name}'. Allowed values: {ALLOWED_NAICS}")

    endpoint_url = "https://frink.apps.renci.org/qlever-geo/sparql"

    query = f"""
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX fio: <http://sawgraph.spatialai.org/v1/fio#>
PREFIX frs: <http://sawgraph.spatialai.org/v1/us-frs#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>

SELECT DISTINCT 
    ?facilityName
    (GROUP_CONCAT(DISTINCT ?industryCode; separator=", ") AS ?industryCodes)
    ?facilityWKT
    ?countyName
    ?stateName
    ?frsId
    ?triId
    ?rcraId
    ?airId
    ?npdesId
    (GROUP_CONCAT(DISTINCT ?envInterestType; separator=", ") AS ?envInterestTypes)
    ?facility
WHERE {{
    ?facility a frs:FRS-Facility ;
              rdfs:label ?facilityName ;
              fio:ofIndustry/rdfs:label ?industryCode ;
              geo:hasGeometry/geo:asWKT ?facilityWKT ;
              kwg-ont:sfWithin ?county .
    ?county rdfs:label ?countyName ;
            kwg-ont:sfWithin ?state .
    ?state rdf:type kwg-ont:AdministrativeRegion_1 ;
           rdfs:label ?stateName .

    FILTER(CONTAINS(LCASE(?stateName), "{state.lower()}"))
    FILTER(CONTAINS(LCASE(?industryCode), "{naics_name.lower()}"))
    FILTER(STRSTARTS(STR(?county), "http://stko-kwg.geog.ucsb.edu/lod/resource/administrativeRegion")) .

    OPTIONAL {{ ?facility frs:hasFRSId ?frsId. }}
    OPTIONAL {{ ?facility frs:hasTRISId ?triId. }}
    OPTIONAL {{ ?facility frs:hasRCRAINFOId ?rcraId. }}
    OPTIONAL {{ ?facility frs:hasAIRId ?airId. }}
    OPTIONAL {{ ?facility frs:hasNPDESId ?npdesId. }}
    OPTIONAL {{ ?facility frs:environmentalInterestType ?envInterestType. }}
}}
GROUP BY ?facility ?facilityName ?facilityWKT ?countyName ?stateName ?industryCode ?frsId ?triId ?rcraId ?airId ?npdesId
LIMIT {limit}
"""    
    # Fetch data
    df = sparql_dataframe.get(endpoint_url, query)

    # Convert WKT to geometry
    df = df.dropna(subset=["facilityWKT"]).copy()
    df["geometry"] = df["facilityWKT"].apply(wkt.loads)
    df = df.drop(columns=["facilityWKT"])

    # Convert to GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")
    # gdf = gdf.drop_duplicates(subset='geometry')  
    return gdf

def _safe_wkt_to_geom(wkt_string):
    try:
        if wkt_string is None:
            return None
        # Some SPARQL clients return a dict / object; ensure string
        if not isinstance(wkt_string, str):
            wkt_string = str(wkt_string)
        return wkt.loads(wkt_string)
    except Exception:
        return None

def load_usda_ars_sites(state=None, pesticide=False):
    """
    Load USDA ARS sites from the sockg SPARQL endpoint into a GeoDataFrame.

    Args:
        state (str|None): optional two-letter US state abbreviation (e.g. 'PA' or 'pa').
        pesticide (bool): if True include aggregated pesticide columns.

    Returns:
        geopandas.GeoDataFrame
    """
    ENDPOINT = "https://idir.uta.edu/sockg_graphdb_v2/repositories/sockg-legacy"
    # validate state
    state_literal = None
    if state is not None:
        state = state.strip()
        if len(state) != 2 or not state.isalpha():
            raise ValueError("state must be a two-letter abbreviation (e.g. 'PA')")
        state_literal = state.upper()

    # Common prefixes
    prefixes = """
PREFIX sockg: <https://idir.uta.edu/sockg-ontology/docs/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
"""

    if pesticide:
        query_body = f"""
SELECT DISTINCT 
       ?siteId ?city ?county ?state ?siteGeometry 
       (GROUP_CONCAT(DISTINCT ?pesticideType; separator=", ") AS ?pesticideTypes)
       (SUM(?totalAmount) AS ?totalPesticideAmount)
       (AVG(?totalAmount) AS ?avgPesticideAmount)
WHERE {{
  # site and geometry
  ?site a sockg:Site ;
        geo:hasGeometry/geo:asWKT ?siteGeometry ;
        sockg:siteId ?siteId ;
        sockg:hasField ?field .

  OPTIONAL {{ ?site sockg:locatedInCity/sockg:cityName ?city. }}
  OPTIONAL {{ ?site sockg:locatedInCounty/sockg:countyName ?county. }}
  OPTIONAL {{ ?site sockg:locatedInState/sockg:stateProvince ?state. }}

  # experimental units -> amendments -> pesticides
  ?expUnit sockg:locatedInField ?field ;
           sockg:hasAmendment ?amendment .
  ?amendment sockg:hasPesticide ?pesticide .

  ?pesticide a sockg:Pesticide ;
             sockg:pesticideActiveIngredientType ?pesticideType ;
             sockg:totalPesticideAmount_kg_per_ha ?totalAmount .
"""
        group_by = "}\nGROUP BY ?siteId ?city ?county ?state ?siteGeometry\nLIMIT " + str(1000)
        query = prefixes + query_body
        if state_literal:
            # append FILTER just before closing WHERE
            query += f'  FILTER(LCASE(STR(?state)) = LCASE("{state_literal}"))\n'
        query += group_by

    else:
        query_body = f"""
SELECT DISTINCT ?siteId ?city ?county ?state ?siteGeometry
WHERE {{
  ?site a sockg:Site ;
        geo:hasGeometry/geo:asWKT ?siteGeometry ;
        sockg:siteId ?siteId .

  OPTIONAL {{ ?site sockg:locatedInCity/sockg:cityName ?city. }}
  OPTIONAL {{ ?site sockg:locatedInCounty/sockg:countyName ?county. }}
  OPTIONAL {{ ?site sockg:locatedInState/sockg:stateProvince ?state. }}
"""
        query = prefixes + query_body
        if state_literal:
            query += f'  FILTER(LCASE(STR(?state)) = LCASE("{state_literal}"))\n'
        query += "}\nLIMIT " + str(100)

    # Run query and build GeoDataFrame
    try:
        df = sparql_dataframe.get(ENDPOINT, query)
    except Exception as e:
        print(f"[load_usda_ars_sites] SPARQL query failed: {e}")
        # return empty GeoDataFrame with expected columns
        if pesticide:
            cols = ["siteId", "city", "county", "state", "siteGeometry",
                    "pesticideTypes", "totalPesticideAmount", "avgPesticideAmount", "geometry"]
        else:
            cols = ["siteId", "city", "county", "state", "siteGeometry", "geometry"]
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

    # If the query returned nothing, return empty GeoDataFrame quickly
    if df is None or df.empty:
        if pesticide:
            cols = ["siteId", "city", "county", "state", "siteGeometry",
                    "pesticideTypes", "totalPesticideAmount", "avgPesticideAmount", "geometry"]
        else:
            cols = ["siteId", "city", "county", "state", "siteGeometry", "geometry"]
        return gpd.GeoDataFrame(columns=cols, geometry="geometry", crs="EPSG:4326")

    # Convert siteGeometry -> geometry column
    if "siteGeometry" in df.columns:
        df["geometry"] = df["siteGeometry"].apply(lambda v: _safe_wkt_to_geom(v))
    else:
        df["geometry"] = None

    # Convert numeric pesticide columns to floats if present
    if pesticide:
        if "totalPesticideAmount" in df.columns:
            df["totalPesticideAmount"] = pd.to_numeric(df["totalPesticideAmount"], errors="coerce")
        if "avgPesticideAmount" in df.columns:
            df["avgPesticideAmount"] = pd.to_numeric(df["avgPesticideAmount"], errors="coerce")

    # Create GeoDataFrame, set CRS (assuming WGS84 lon/lat)
    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs="EPSG:4326")

    return gdf


def load_military_bases(where: str = "1=1", bbox: Optional[List[float]] = None) -> gpd.GeoDataFrame:
    """Load watershed boundary dataset using concurrent fetching."""
    url = "https://services.arcgis.com/xOi1kZaI0eWDREZv/arcgis/rest/services/NTAD_Military_Bases/FeatureServer/0"
    
    loader = ArcGISFeatureLoader(
        url=url,
        batch_size=100,
        max_workers=4,
        max_retries=3
    )
    
    gdf = loader.load_features(where=where, bbox=bbox)
    gdf.title = "All Basins"
    return gdf


