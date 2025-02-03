from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


#####################################################################
# Implement the Request Planer
def get_request_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
                    You are an expert in the WEN-OKN knowledge system, which answers 
                    one question or returns data for one entity type at a time. For 
                    example, you can return dams or earthquakes. Your task is to extract 
                    a list of atomic requests from the user's question based on the entity
                    types requested to fetch. Each atomic request must be executable 
                    independently of the others.

                    Example 1:   
                    Original Question: "First find Scioto River and all dams on it."
                    
                    Atomic Requests:
                        "Find Scioto River."
                        "Find all dams on Scioto River."
                        
                    Example 2:
                    Original Question: "First find Scioto River, then find all dams 
                    on this river. Also find all counties these dams locate."
                    Atomic Requests:
                        "Find Scioto River."
                        "Find all dams on Scioto River."
                        "Find all counties where dams on Scioto River are located."
                        
                    Example 3:
                    Original Question: "Find all counties Scioto River flows through."
                    Atomic Request:
                        "Find all counties Scioto River flows through."
                    
                    Example 4:
                    Original Question: "Find all dams on Scioto River."
                    Atomic Request:
                        "Find all dams on Scioto River."

                    Example 5:
                    Original Question: "Find all counties both Scioto River and Ohio River flow through."
                    Atomic Request:
                        "Find all counties both Scioto River and Ohio River flow through."

                    Example 6:
                    Original Question: "Find all dams located upstream of power station dpjc6wtthc32 along the Muskingum River."
                    Atomic Request:
                        "Find all dams located upstream of power station dpjc6wtthc32 along the Muskingum River."
                    
                    Task
                    Divide the user's question into atomic requests based on the entity types 
                    mentioned. Each atomic request should be in its original form from the 
                    question and should not contain any pronouns like "it" or "its" or "this" or
                    "these" or "those" to ensure independent handling.
                    
                    Input
                    User Question:
                    {question}
                    
                    Output
                    Return your answer in JSON format with a list of atomic requests under the key 
                    "requests" without preamble or explanation.

                    <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result


#####################################################################
# Implement the Aggregation Planer
def get_aggregation_plan_2(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

You are an expert in query analysis. Extract key components from the given user request, which describes an aggregation query.

Extraction Rules
    - Grouping Object: The entity used for grouping (e.g., county, state).
        * If not explicitly stated, infer the most reasonable entity from the query.
        * If multiple grouping entities exist, choose the most specific one.
    - Summarizing Object: The entity being aggregated (e.g., river, hospital).
        * If not explicitly stated, infer the entity that is being counted, summed, or aggregated.
        * Never use "aggregation" as a placeholder—always extract a meaningful entity.
    - Association Conditions: The relationship between the grouping and summarizing objects.
        * If missing, infer a reasonable relationship (e.g., "river flows through county").
    - Aggregation Function: The mathematical/statistical operation applied (e.g., COUNT, SUM, ARGMAX).
        * Always return in uppercase.
        * If missing, infer the most logical function based on the query.
    - Preconditions: Filters applied before aggregation (e.g., "county is in Ohio").
        * If none exist, return null.
    - Postconditions: Filters applied after aggregation (e.g., "COUNT > 5").
        * If none exist, return null.

Example Extraction
User Request: "For each county in Ohio, find the number of rivers flowing through the county."

This request can be defined as the following query:

    SELECT county, COUNT(river) AS river_count   
    FROM county, river
    WHERE county in 'Ohio'  
      AND river INTERSECTS county  
    GROUP BY county
    
The object used in "GROUP BY" is the grouping object. The object used to apply the aggregation function COUNT is the summarizing object.

Extraction Output:
{{
  "pseudo_query": ....,
  "grouping_object": "county",
  "summarizing_object": "river",
  "association_conditions": "river flows through county",
  "aggregation_function": "COUNT",
  "preconditions": "county in Ohio state",
  "postconditions": null
}}

Strict Guidelines for Extraction
    - Do not return generic placeholders like "aggregation".
    - Ensure that "grouping_object" and "summarizing_object" are never null.
    - If the user request is ambiguous, infer the most logical structure.
    - Only return a JSON object. No explanations, no additional text.
    - For association conditions, construct a meaningful relationship between the grouping and summarizing objects.

User Request:
{question}



         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result


#####################################################################
# Implement the Aggregation Planer
def get_aggregation_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 

You are an expert in query analysis. Your task is to convert the user request into a query in the following pseduo query format.

Example 1.1  "How many rivers flow through each county in Ohio?"

SELECT county.name, COUNT(river.id) AS river_count
FROM county, river
WHERE county.state = 'Ohio' 
  AND river.geometry INTERSECTS county.geometry 
GROUP BY county.name
     
Example 1.2  "How many dams are in each county of Ohio?"

SELECT county.name, COUNT(dam.id) AS dam_count
FROM county, dam
WHERE dam.geometry INSIDE county.geometry  
GROUP BY county.name

Example 1.3  "How many coal mines are in each basin?"

SELECT basin.name, COUNT(coalmine.id) AS coalmine_count
FROM basin, coalmine
WHERE coalmine.geometry INSIDE basin.geometry  
GROUP BY basin.name

Example 2.1  "What is the total power generation capacity of power plants in each county?"

SELECT county.name, SUM(powerplant.capacity) AS total_capacity
FROM county, powerplant
WHERE powerplant.geometry INSIDE county.geometry  
GROUP BY county.name
 
Example 3.1  "What is the longest river in each state?" (Object-centric aggregation)

SELECT state.name, 
  ARGMAX(river.name, SPATIAL_LENGTH(INTERSECTION(river.geometry, state.geometry))) AS longest_river
FROM state, river
WHERE river.geometry INTERSECTS state.geometry  -- Spatial overlap
GROUP BY state.name
	 
Example 3.2 "Which county has the highest number of hospitals?"
     
SELECT county.name, COUNT(hospital.id) AS hospital_count
FROM county, hospital
WHERE hospital.geometry INSIDE county.geometry
GROUP BY county.name
ORDER BY hospital_count DESC
LIMIT 1  -- Return the county with the maximum count
  
Example 4.1 "What is the average dam height per watershed?"

SELECT watershed.name, AVG(dam.height) AS avg_dam_height
FROM watershed, dam
WHERE dam.geometry INSIDE watershed.geometry
GROUP BY watershed.name

Example 4.2 "What is the average discharge of rivers in each basin?"

SELECT basin.name, AVG(river.discharge) AS avg_discharge
FROM basin, river
WHERE river.geometry INSIDE basin.geometry
GROUP BY basin.name

Example 5.1 "List all counties with more than 5 hospitals."

SELECT county.name, COUNT(hospital.id) AS hospital_count
FROM county, hospital
WHERE hospital.geometry INSIDE county.geometry
GROUP BY county.name
HAVING hospital_count > 5  -- Post-aggregation filter

Example 5.2 "Find all states where the total coal mine output exceeds 1 million tons."

SELECT state.name, SUM(coalmine.output) AS total_output
FROM state, coalmine
WHERE coalmine.geometry INSIDE state.geometry
GROUP BY state.name
HAVING total_output > 1000000

Example 6.1 "Which river has the highest number of dams?"

SELECT river.name, COUNT(dam.id) AS dam_count
FROM river, dam
WHERE dam.geometry INTERSECTS river.geometry  -- Dams along the river
GROUP BY river.name
ORDER BY dam_count DESC
LIMIT 1  -- Return the river with the maximum dam count

Example 6.2 "Which watershed has the highest total coal mine production?"

SELECT watershed.name, SUM(coalmine.production) AS total_production
FROM watershed, coalmine
WHERE coalmine.geometry INSIDE watershed.geometry
GROUP BY watershed.name
ORDER BY total_production DESC
LIMIT 1  -- Return the watershed with the maximum production


User Request:
{question}

         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )

    formatted_prompt = prompt.format(question=question)

    # Print the exact text sent to the LLM
    print("=== Prompt Sent to LLM ===")
    print(formatted_prompt)
    print("==========================")
    return formatted_prompt
	
    question_planer = prompt | llm | StrOutputParser()
    result = question_planer.invoke({"question": question})
    return result
