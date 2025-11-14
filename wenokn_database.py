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
from shapely.ops import transform, substring
import shapely.geometry
import shapely.ops
from shapely.geometry import Point, LineString, MultiLineString
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
}}
LIMIT 1
"""
    logger.info(query)
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
WHERE {{
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
}}
LIMIT 2000
"""
    logger.info(query)
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_dams_in_states(state_names: list[str]) -> gpd.GeoDataFrame:
    cleaned_states = []
    for name in state_names:
        if not name:
            continue
        name = name.strip()
        if name.lower().endswith(" state"):
            name = name[:-6].strip()
        cleaned_states.append(name.lower())

    values_block = "\n        ".join(f'"{s}"' for s in cleaned_states)

    query = f"""
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?damName ?damGeometry
WHERE {{
    ?dam schema:provider "https://nid.usace.army.mil"^^<https://schema.org/url> ;
         schema:name ?damName ;
         geo:hasGeometry/geo:asWKT ?damGeometry .
    FILTER(STRSTARTS(STR(?dam), "https://geoconnex.us/ref/dams/"))

    ?state rdf:type <http://stko-kwg.geog.ucsb.edu/lod/ontology/AdministrativeRegion_1> ;
           rdfs:label ?stateName ;
           geo:hasGeometry/geo:asWKT ?stateGeometry .
    FILTER(STRSTARTS(STR(?state), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))

    VALUES ?inputState {{
        {values_block}
    }}
    FILTER(STRSTARTS(LCASE(STR(?stateName)), LCASE(?inputState)))

    FILTER(geof:sfContains(?stateGeometry, ?damGeometry))
}}
"""
    logger.info(query)
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_dams_in_counties(county_names: list[str]) -> gpd.GeoDataFrame:
    """
    Build a SPARQL query to find all dams inside the given counties.
    County names are used exactly as provided, without cleaning.
    """
    if not county_names:
        raise ValueError("county_names list cannot be empty.")

    # Build the VALUES clause
    values_rows = "\n".join(f'("{name}")' for name in county_names)
    values_clause = f"VALUES (?inputCounty) {{\n{values_rows}\n}}"

    # Full SPARQL query
    query = f"""
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>

SELECT ?damName ?damGeometry ?countyName
WHERE {{
    # Dams from NID
    ?dam schema:provider "https://nid.usace.army.mil"^^<https://schema.org/url>;
         schema:name ?damName ;
         geo:hasGeometry/geo:asWKT ?damGeometry .
    FILTER(STRSTARTS(STR(?dam), "https://geoconnex.us/ref/dams/"))

    # Counties
    ?county rdf:type kwg-ont:AdministrativeRegion_2 ;
            rdfs:label ?countyName ;
            geo:hasGeometry/geo:asWKT ?countyGeometry .
    FILTER(STRSTARTS(STR(?county), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))

    # Inject county names exactly as provided
    {values_clause}

    FILTER(STR(?countyName) = ?inputCounty)

    # Spatial containment
    FILTER(geof:sfContains(?countyGeometry, ?damGeometry))
}}
"""
    logger.info(query)
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_counties_river_flows_through(river_name) -> gpd.GeoDataFrame:    
        
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT ?countyName ?countyGeometry 
WHERE {{
  ?river a hyf:HY_FlowPath ;
         a hyf:HY_WaterBody ;
         a schema:Place ;
         schema:name ?riverName ;
         geo:hasGeometry/geo:asWKT ?riverGeometry .
  FILTER(LCASE(?riverName) = LCASE("{river_name}")) .
  
  ?county rdf:type kwg-ont:AdministrativeRegion_2 ;
          rdfs:label ?countyName ;
          geo:hasGeometry/geo:asWKT ?countyGeometry .
  FILTER(STRSTARTS(STR(?county), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))
  
  FILTER(geof:sfIntersects(?riverGeometry, ?countyGeometry))
}}
LIMIT 200
"""
    logger.info(query)
    return get_gdf_from_sparql(query)
    

#-----------------------------------------------------
def load_states_river_flows_through(river_name) -> gpd.GeoDataFrame:    
        
    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT ?stateName ?stateGeometry 
WHERE {{
  ?river a hyf:HY_FlowPath ;
         a hyf:HY_WaterBody ;
         a schema:Place ;
         schema:name ?riverName ;
         geo:hasGeometry/geo:asWKT ?riverGeometry .
  FILTER(LCASE(?riverName) = LCASE("{river_name}")) .
  
  ?state rdf:type kwg-ont:AdministrativeRegion_1 ;
          rdfs:label ?stateName ;
          geo:hasGeometry/geo:asWKT ?stateGeometry .
  FILTER(STRSTARTS(STR(?state), "http://stko-kwg.geog.ucsb.edu/lod/resource/"))
  FILTER(strlen(?stateName) > 2)

  FILTER(geof:sfIntersects(?riverGeometry, ?stateGeometry))
}}
LIMIT 200
"""
    logger.info(query)
    return get_gdf_from_sparql(query)


#-----------------------------------------------------
def load_counties_rivers_flow_through_all(river_names: list[str]) -> gpd.GeoDataFrame:
    """
    Return counties that all given rivers flow through (intersection of coverage),
    using subqueries to avoid QLever planner crashes.
    """

    subqueries = []
    vars_decl = []

    for i, river in enumerate(river_names):
        river_geom_var = f"?riverGeometry{i}"
        vars_decl.append(river_geom_var)
        subqueries.append(f"""
  {{
    SELECT DISTINCT {river_geom_var}
    WHERE {{
      ?river{i} a hyf:HY_FlowPath ;
                a hyf:HY_WaterBody ;
                a schema:Place ;
                schema:name ?riverName{i} ;
                geo:hasGeometry/geo:asWKT {river_geom_var} .
      FILTER(LCASE(?riverName{i}) = LCASE("{river}")) .
    }}
  }}
""")

    # Build intersection filters
    intersection_filters = "\n  ".join(
        [f"FILTER(geof:sfIntersects(?countyGeometry, {var})) ." for var in vars_decl]
    )

    query = f"""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX geo: <http://www.opengis.net/ont/geosparql#>
PREFIX geof: <http://www.opengis.net/def/function/geosparql/>
PREFIX kwg-ont: <http://stko-kwg.geog.ucsb.edu/lod/ontology/>
PREFIX kwgr: <http://stko-kwg.geog.ucsb.edu/lod/resource/>
PREFIX hyf: <https://www.opengis.net/def/schema/hy_features/hyf/>
PREFIX schema: <https://schema.org/>

SELECT DISTINCT ?countyName ?countyGeometry
WHERE {{
{subqueries[0]}
{subqueries[1] if len(subqueries) > 1 else ''}

  ?county rdf:type kwg-ont:AdministrativeRegion_2 ;
          rdfs:label ?countyName ;
          geo:hasGeometry/geo:asWKT ?countyGeometry .
  FILTER(STRSTARTS(STR(?county), STR(kwgr:))) .

  {intersection_filters}
}}
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
Please make sure the state name is a valid and fix possible typos.

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

If the user's question is to find all counties a river flows through (for example, Find all counties Ohio River flows through), 
return the following code:
    river_name = "Ohio River"
    gdf = load_counties_river_flows_through(river_name)  
    gdf.title = "All counties Ohio River flows through"

If the user's question is to find all states a river flows through (for example, Find all states Ohio River flows through), 
return the following code:
    river_name = "Ohio River"
    gdf = load_states_river_flows_through(river_name)  
    gdf.title = "All states Ohio River flows through"

If the user's question is to find all counties multiple rivers flow through (for example, Find all counties Ohio River and Muskingum River flow through), 
return the following code:
    river_names = [ "Ohio River", "Muskingum River" ]
    gdf = load_counties_rivers_flow_through_all(river_names)  
    gdf.title = "All counties Ohio River and Muskingum River flow through"

If the user's question is to find all downstream counties of a river from a county (for example, find all downstream 
counties of the Scioto River from the Ross County), you can return the following code:
    river_name = "Scioto River"
    county_name = "Ross County"
    river_gdf = load_river_by_name(river_name)
    county_gdf = load_county_by_name(county_name)
    river_geom = river_gdf.geometry.iloc[0]
    county_geom = county_gdf.geometry.iloc[0]
    intersection_geom = river_geom.intersection(county_geom)
    if intersection_geom.is_empty:
        raise ValueError(f"The Scioto river does not pass through the Ross county.")
    from shapely.geometry import Point, LineString, MultiLineString
    def extract_downstream_point(intersection, river_line):
        if isinstance(intersection, Point):
            return intersection
        if isinstance(intersection, LineString):
            candidates = [Point(intersection.coords[0]), Point(intersection.coords[-1])]
            downstream_end = Point(river_line.coords[-1])
            return min(candidates, key=lambda p: p.distance(downstream_end))
        if isinstance(intersection, MultiLineString):
            downstream_end = Point(river_line.coords[-1])
            endpoints = []
            for seg in intersection.geoms:
                endpoints.append(Point(seg.coords[0]))
                endpoints.append(Point(seg.coords[-1]))
            return min(endpoints, key=lambda p: p.distance(downstream_end))
        raise ValueError("Unexpected intersection geometry")
    entry_point = extract_downstream_point(intersection_geom, river_geom)
    distance_on_river = river_geom.project(entry_point)
    from shapely.ops import substring
    downstream_segment = substring(river_geom, distance_on_river, river_geom.length)
    counties_gdf = load_counties_river_flows_through(river_name)
    gdf = counties_gdf[counties_gdf.intersects(downstream_segment)]
    gdf.title = f"find all downstream counties of the Scioto River from the Ross County"

If the user's question requires multiple steps (for example, 
    "Find all rivers that pass the counties Scioto River passes" 
    or 
    "Find all rivers that pass all downstream counties of the Scioto River from Ross County"), 
you must compose previously defined functions to solve it.

Example pattern:
    1. Identify the required base entities (river_name, county_name, state_name, etc.)
    2. Call the appropriate load_* functions to get intermediate GeoDataFrames.
    3. Perform necessary spatial operations (intersection, union, filtering).
    4. Construct the final GeoDataFrame and set a meaningful title.
    5. Do not print anything.

For example: "Find all rivers that pass the counties Scioto River passes"
Return code like:
    river_name = "Scioto River"
    counties_gdf = load_counties_river_flows_through(river_name)

    river_sets = []
    for _, row in counties_gdf.iterrows():
        rivers = load_rivers_in_county(row["countyName"])
        river_sets.append(rivers)

    import geopandas as gpd
    gdf = gpd.GeoDataFrame( 
        pd.concat(river_sets, ignore_index=True)
    ).drop_duplicates(subset=["riverName"])
    gdf.title = "All rivers that pass the counties Scioto River passes"

If the user's question is to find all dams in some states (for example, Find all dams in the Ohio State), 
return the following code:
    state_names = [ "Ohio State" ]
    gdf = load_dams_in_states(state_names)  
    gdf.title = "All dams in the Ohio State"

If the user's question is to find all dams in some counties (for example, Find all dams in the Ross County), 
return the following code:
    county_names = [ "Ross county" ]
    gdf = load_dams_in_counties(county_names)  
    gdf.title = "All dams in the Ross County"


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
        
  
