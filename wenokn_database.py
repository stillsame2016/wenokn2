import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


def process_wenokn_request(llm, user_input, chat_container):
    prompt = PromptTemplate(
        template="""
<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
Your task is to return valid Python code based on the user's question.

If the user's question is to look up a river by name, return the following code:
    gdf = load_river_by_name(river_name)
    gdf.title = river_name

[ Question ]
{ question }
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    df_code_chain = prompt | llm | StrOutputParser() 
    return df_code_chain.invoke({"question": user_input})
        
  
