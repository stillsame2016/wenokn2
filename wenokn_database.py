import geopandas as gpd
import pandas as pd
import requests
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser


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
        
  
