import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

import io
import pyproj
import json
from shapely.geometry import Point, box, LineString, MultiLineString
from shapely.ops import transform
import shapely.geometry
import shapely.ops
import numpy as np
import concurrent.futures
import time
from typing import Optional, List, Dict, Any, Union
import logging
from shapely import wkt
import sparql_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


from shapely import wkt
import geopandas as gpd
import sparql_dataframe

def get_gdf_from_sparql(query):
    endpoint_url = "https://frink.apps.renci.org/federation/sparql"
    df = sparql_dataframe.get(endpoint_url, query)

    if df.empty:
        return gpd.GeoDataFrame(columns=["geometry"], crs="EPSG:4326")

    # Identify the WKT column automatically
    wkt_col = None
    for col in df.columns:
        if df[col].astype(str).str.startswith(("POINT", "LINESTRING", "POLYGON", "MULTIPOLYGON")).any():
            wkt_col = col
            break

    if wkt_col is None:
        raise ValueError("No WKT geometry column found in SPARQL result.")

    # Drop missing geometries
    df = df.dropna(subset=[wkt_col]).copy()

    # Convert WKT to shapely geometries (keep same column name)
    # df[wkt_col] = df[wkt_col].apply(wkt.loads)
    df['geometry'] = df[wkt_col].apply(wkt.loads)

    # Create GeoDataFrame using that same column name as geometry
    # gdf = gpd.GeoDataFrame(df, geometry=wkt_col, crs="EPSG:4326")
    gdf = gpd.GeoDataFrame(df, geometry='geometry', crs="EPSG:4326")

    return gdf


#-----------------------------------------------------
def load_river_by_name(river_name) -> gpd.GeoDataFrame:
    query = f"""
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>

SELECT DISTINCT ?riverName ?riverGeometry 
WHERE {{
  ?river a hyf:HY_FlowPath ;
         a hyf:HY_WaterBody ;
         a schema:Place ;
         schema:name ?riverName ;
         geo:hasGeometry/geo:asWKT ?riverGeometry .
  FILTER(LCASE(?riverName) = LCASE("{river_name}")) .
}}
ORDER BY DESC(STRLEN(?riverGeometry))
LIMIT 1
"""
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_county_by_name(county_name) -> gpd.GeoDataFrame:
    query = f"""
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?countyName ?countyGeometry
WHERE {{
  ?county rdf:type <http://stko-kwg.geog.ucsb.edu/lod/ontology/AdministrativeRegion_2> ;
          rdfs:label ?countyName ;
          geo:hasGeometry/geo:asWKT ?countyGeometry .
  FILTER(STRSTARTS(STR(?county), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))
  BIND(LCASE("{county_name}") AS ?inputCounty)
  FILTER(STRSTARTS(LCASE(STR(?countyName)), ?inputCounty))
}}
LIMIT 1
"""
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_state_by_name(state_name) -> gpd.GeoDataFrame:
    if state_name and state_name.lower().endswith(" state"):
        state_name = state_name[:-6]
        
    query = f"""
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT DISTINCT ?stateName ?stateGeometry
WHERE {{
  ?state rdf:type <http://stko-kwg.geog.ucsb.edu/lod/ontology/AdministrativeRegion_1> ;
          rdfs:label ?stateName ;
          geo:hasGeometry/geo:asWKT ?stateGeometry .
  FILTER(STRSTARTS(STR(?state), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))
  BIND(LCASE("{state_name}") AS ?inputState)
  FILTER(STRSTARTS(LCASE(STR(?stateName)), ?inputState))

  ?county rdf:type <http://stko-kwg.geog.ucsb.edu/lod/ontology/AdministrativeRegion_2> ;
          rdfs:label ?countyName ;
          geo:hasGeometry/geo:asWKT ?countyGeometry .
  FILTER(STRSTARTS(STR(?county), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))

  FILTER (geof:sfIntersects(?countyGeometry, ?stateGeometry)) . 

}}
LIMIT 1
"""
    return get_gdf_from_sparql(query)
    

#-----------------------------------------------------
def load_counties_in_state(state_name) -> gpd.GeoDataFrame:
    if state_name and state_name.lower().endswith(" state"):
        state_name = state_name[:-6]
        
    query = f"""
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?countyName ?countyGeometry
WHERE {{
  ?county rdf:type <http://stko-kwg.geog.ucsb.edu/lod/ontology/AdministrativeRegion_2> ;
          rdfs:label ?countyName ;
          geo:hasGeometry/geo:asWKT ?countyGeometry .
  FILTER(STRSTARTS(STR(?county), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))
  BIND(LCASE("{state_name}") AS ?inputState)
  FILTER(STRENDS(LCASE(STR(?countyName)), ?inputState))
}}
LIMIT 200
"""
    logger.info(query)
    return get_gdf_from_sparql(query)
    

#-----------------------------------------------------
def load_neighboring_counties(county_name) -> gpd.GeoDataFrame:        
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>

SELECT DISTINCT ?neighborCountyName ?neighborCountyGeometry
WHERE {{
  ?neighborCounty rdf:type kwg-ont:AdministrativeRegion_2 ;
                  rdfs:label ?neighborCountyName ;
                  geo:hasGeometry/geo:asWKT ?neighborCountyGeometry .
  FILTER(STRSTARTS(STR(?neighborCounty), STR(kwgr:)))  # cleaner URI prefix check

  FILTER EXISTS {{
    ?county rdf:type kwg-ont:AdministrativeRegion_2 ;
                    rdfs:label ?countyName ;
                    kwg-ont:sfOverlaps ?s2cell .
    FILTER(STRSTARTS(STR(?county), STR(kwgr:)))
    FILTER(STRSTARTS(LCASE(?countyName), LCASE("{county_name}")))

    # Shared S2 cell constraint
    ?neighborCounty kwg-ont:sfOverlaps ?s2cell .
    ?s2cell rdf:type kwg-ont:S2Cell_Level13 .

    FILTER(?neighborCounty != ?county)
  }}
}}
LIMIT 100
"""
    logger.info(query)
    return get_gdf_from_sparql(query)
    

#-----------------------------------------------------
def load_neighboring_states(state_name) -> gpd.GeoDataFrame:     
    if state_name and state_name.lower().endswith(" state"):
        state_name = state_name[:-6]
        
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>

SELECT DISTINCT ?neighborStateName ?neighborStateGeometry
WHERE {{
  # --- Step 1: Compute the longest label for each state ---
  {{
    SELECT ?neighborState (MAX(STRLEN(STR(?label))) AS ?maxLen)
    WHERE {{
      ?neighborState rdf:type kwg-ont:AdministrativeRegion_1 ;
                     rdfs:label ?label .
      FILTER(STRSTARTS(STR(?neighborState), STR(kwgr:)))
    }}
    GROUP BY ?neighborState
  }}

  # --- Step 2: Join to get the full label and geometry ---
  ?neighborState rdf:type kwg-ont:AdministrativeRegion_1 ;
                 rdfs:label ?neighborStateName ;
                 geo:hasGeometry/geo:asWKT ?neighborStateGeometry .
  FILTER(STRLEN(STR(?neighborStateName)) = ?maxLen)

  # --- Step 3: Check adjacency via shared S2 cells ---
  FILTER EXISTS {{
    ?state rdf:type kwg-ont:AdministrativeRegion_1 ;
           rdfs:label ?stateLabel ;
           kwg-ont:sfOverlaps ?s2cell .
    FILTER(STRSTARTS(STR(?state), STR(kwgr:)))
    FILTER(CONTAINS(LCASE(?stateLabel), LCASE("{state_name}")))

    ?neighborState kwg-ont:sfOverlaps ?s2cell .
    ?s2cell rdf:type kwg-ont:S2Cell_Level13 .

    FILTER(?neighborState != ?state)
  }}
}}
ORDER BY ?neighborStateName
LIMIT 100
"""
    logger.info(query)
    return get_gdf_from_sparql(query)
    

#-----------------------------------------------------
def load_rivers_in_county(county_name) -> gpd.GeoDataFrame:        
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT ?riverName ?riverGeometry
WHERE {{
    ?county rdf:type kwg-ont:AdministrativeRegion_2 ;
                    rdfs:label ?countyName ;
                    geo:hasGeometry/geo:asWKT ?countyGeometry.
    FILTER(STRSTARTS(STR(?county), STR(kwgr:)))
    FILTER(STRSTARTS(LCASE(?countyName), LCASE("{county_name}")))

  ?river a hyf:HY_FlowPath ;
         a hyf:HY_WaterBody ;
         a schema:Place ;
         schema:name ?riverName ;
         geo:hasGeometry/geo:asWKT ?riverGeometry .
   
   FILTER(geof:sfIntersects(?riverGeometry, ?countyGeometry)) .
}}
LIMIT 300
"""
    logger.info(query)
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_rivers_in_state(state_name) -> gpd.GeoDataFrame:    
    if state_name and state_name.lower().endswith(" state"):
        state_name = state_name[:-6]
        
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT ?riverName ?riverGeometry
WHERE {
  ?state rdf:type kwg-ont:AdministrativeRegion_1 ;
         rdfs:label ?stateName ;
         geo:hasGeometry/geo:asWKT ?stateGeometry .
  FILTER(STRSTARTS(STR(?state), STR(kwgr:)))
  FILTER(STRSTARTS(LCASE(?stateName), LCASE("{state_name}")))

  ?river a hyf:HY_FlowPath ;
         a hyf:HY_WaterBody ;
         a schema:Place ;
         schema:name ?riverName ;
         geo:hasGeometry/geo:asWKT ?riverGeometry .

  FILTER(geof:sfIntersects(?riverGeometry, ?stateGeometry))
  FILTER(BOUND(?riverName) && STRLEN(LCASE(STR(?riverName))) > 0)
}
LIMIT 2000
"""
    logger.info(query)
    return get_gdf_from_sparql(query)
    

def process_wenokn_request(llm, user_input, chat_container):
    prompt = PromptTemplate(
        template="""
Your task is to return valid Python code based on the user's question.

If the user's question is to look up a river by name, return the following code:
    river_name = ...
    gdf = load_river_by_name(river_name)   
    gdf.title = river_name

If the user's question is to look up a county by name, return the following code:
    county_name = ...
    gdf = load_county_by_name(county_name)  
    gdf.title = county_name

If the user's question is to look up a state by name, return the following code:
    state_name = ...
    gdf = load_state_by_name(state_name)  
    gdf.title = state_name

If the user's question is to find all counties in a state (for example, Find all counties in California), 
return the following code:
    state_name = "California"
    gdf = load_counties_in_state(state_name)  
    gdf.title = "All counties in California"

If the user's question is to find all neighboring counties of a county (for example, Find all neighboring counties of Ross county), 
return the following code:
    county_name = "Ross county"
    gdf = load_neighboring_counties(county_name)  
    gdf.title = "All neighboring counties of Ross county"

If the user's question is to find all neighboring states of a state (for example, Find all neighboring states of Ohio state), 
return the following code:
    state_name = "Ohio State"
    gdf = load_neighboring_states(state_name)  
    gdf.title = "All neighboring states of Ohio State"

If the user's question is to find all rivers flows through a county (for example, Find all rivers in Ross county), 
return the following code:
    county_name = "Ross county"
    gdf = load_rivers_in_county(county_name)  
    gdf.title = "All rivers in Ross county"

If the user's question is to find all rivers flows through a state (for example, Find all rivers in Ohio state), 
return the following code:
    state_name = "Ohio State"
    gdf = load_rivers_in_state(state_name)  
    gdf.title = "All rivers in Ohio State"

Otherwise return the following code:
    raise ValueError("Don't know how to process the request")

[ Question ]
The following is the question from the user:
{question}

Don't include any print statement. Don't add ``` around the code.
        """,
        input_variables=["question"],
    )
    df_code_chain = prompt | llm | StrOutputParser() 
    return df_code_chain.invoke({"question": user_input})
        
  
