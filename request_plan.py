from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


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
def get_aggregation_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
        
        An aggregation request may involve 5 core components:
            1) Grouping Object: Entities to partition data by (e.g., counties, basins).
            2) Summarizing Object: Entities to aggregate (e.g., rivers, dams).
            3) Association Conditions: Relationships between grouping and summarizing objects (e.g., spatial containment, spatail intersection).
            4) Aggregation Function: Operations like COUNT, SUM, MAX, AVG, or ARGMAX (for object-centric results).
            5) Pre-/Post-Conditions: Filters applied before/after aggregation (e.g., counties in Ohio State, result thresholds).
        
        Your task is to extract the Grouping Object, the Summarizing Object, the Association Conditions, the Aggregation Function and 
        Preconditions and Postconditions from the input request.

        [ Example 1 ]
            For the request: "Find the number of rivers flow through each county in Ohio."  
            This request is equivalent to the following pseudo query:
                SELECT county.name, COUNT(river.id) AS river_count
                  FROM county, river
                 WHERE county.state = 'Ohio' 
                   AND river.geometry INTERSECTS county.geometry 
                 GROUP BY county.name
            You can return the following JSON string:   
               {{
                   "grouping_object": "county",
                   "summarizing_object": "river",
                   "association_conditions:" "river flows through county",
                   "aggregation_function": "count",
                   "preconditions": "county in Ohio State"
                }}
                        
         User Request:
             {question}
                    
         Return your answer in JSON format without preamble or explanation.

         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result
