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

You are an expert in query analysis. Your task is to extract and structure key components from a user’s aggregation query into a JSON object.

Identify the following:

Grouping Object: The entity used to group data (e.g., county, state). This must be present.
Summarizing Object: The entity being aggregated (e.g., river, hospital). This must be present.
Association Conditions: The explicit or implied relationship between the grouping and summarizing objects (e.g., "river flows through county").
Aggregation Function: The mathematical/statistical operation applied (e.g., COUNT, SUM, ARGMAX(length)).
Preconditions: Any filtering applied before aggregation (e.g., "county is in Ohio"). If none, return null.
Postconditions: Any filtering applied after aggregation (e.g., "COUNT > 5"). If none, return null.

Example Extraction
User Request:
"Find the number of rivers flowing through each county in Ohio."

Extraction Output:
{{
  "grouping_object": "county",
  "summarizing_object": "river",
  "association_conditions": "river flows through county",
  "aggregation_function": "COUNT",
  "preconditions": "county is in Ohio",
  "postconditions": null
}}

Instructions:
Return only a valid JSON object (no extra text).
"grouping_object" and "summarizing_object" cannot be null. If not explicit, infer from context.
"aggregation_function" should always be uppercase (e.g., COUNT, SUM).
If preconditions or postconditions are missing, explicitly return null.

User Request:
{question}




         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result
