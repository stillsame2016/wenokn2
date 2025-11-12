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
    df[wkt_col] = df[wkt_col].apply(wkt.loads)

    # Create GeoDataFrame using that same column name as geometry
    gdf = gpd.GeoDataFrame(df, geometry=wkt_col, crs="EPSG:4326")

    return gdf


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


def process_wenokn_request(llm, user_input, chat_container):
    prompt = PromptTemplate(
        template="""
Your task is to return valid Python code based on the user's question.

If the user's question is to look up a river by name, return the following code:
    gdf = load_river_by_name(river_name)
    gdf.title = river_name

[ Question ]
The following is the question from the user:
{question}

Don't include any print statement. Don't add ``` around the code.
        """,
        input_variables=["question"],
    )
    df_code_chain = prompt | llm | StrOutputParser() 
    return df_code_chain.invoke({"question": user_input})
        
  
