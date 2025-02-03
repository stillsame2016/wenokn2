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

        Your task is to identify the following components from the user request which is an aggregation query:
            1. Grouping Object (e.g., county, state): Entities to group data by.
            2. Summarizing Object (e.g., river, hospital): Entities being aggregated.
            3. Association Conditions: Relationships between grouping/summarizing objects (e.g., flows through, located in).
            4. Aggregation Function (e.g., COUNT, ARGMAX(length)): Operation to apply.
            5. Preconditions: Filters applied before aggregation (e.g., county is in Ohio).
            6. Postconditions: Filters applied after aggregation (e.g., COUNT > 5).
        Note that the Grouping Object and Summarizing Object are not null.
                
        Examples
        
        Example 1: For the request "Find the number of rivers flowing through each county in Ohio",
        You can return
        {{ 
          "grouping_object": "county",  
          "summarizing_object": "river",  
          "association_conditions": "river flows through county",  
          "aggregation_function": "COUNT",  
          "preconditions": "county is in Ohio",  
          "postconditions": null  
        }}  
        
        Example 2: For the request "List counties with more than 5 hospitals" , 
        You can return
        {{  
          "grouping_object": "county",  
          "summarizing_object": "hospital",  
          "association_conditions": "hospital is located in county",  
          "aggregation_function": "COUNT",  
          "preconditions": null,  
          "postconditions": "COUNT > 5"  
        }} 
        
        Example 3: For the request, "What is the longest river in each state?"  
        You can return
        {{  
          "grouping_object": "state",  
          "summarizing_object": "river",  
          "association_conditions": "river is located in state",  
          "aggregation_function": "ARGMAX(length)",  
          "preconditions": null,  
          "postconditions": null  
        }}  
        
        Instructions
            Return only a valid JSON object with the keys above.
            Use null for missing pre/postconditions.
            Use uppercase for aggregation functions (e.g., COUNT, not count).
            "grouping_object" and "summarizing_object" can't be null.

        User Request:
        {question}

        

         <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result
