from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
# Implement the Request Refiner
def get_refined_question(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>You are an expert of the 
            WEN-OKN knowledge database. You also have general knowledge. 
            
            The following is a question the user is asking:
    
            [--- Start ---]
            {question}
            [--- End ---]
    
            Your main job is to determine if the user is requesting for data in the scope of the WEN-OKN 
            knowledge database.
    
            If they are requesting for data in the scope of the WEN-OKN knowledge database, then extract 
            the request from the user's input. Rephrase the user's request in a formal way. Remove all 
            adjectives like "beautiful" or "pretty". Remove the terms like "Please" etc. Use the format 
            like "Find ...". The place name must be retained if it is mentioned in the request. If a place 
            name may be both a county or a state, for example, Ohio, then use Ohio State. Keep the the 
            number of user requested entities.
    
            Please answer with a valid JSON string only without any preamble or explanation, including the 
            following three fields:
    
            The boolean field "is_request_data" is true if the user is requesting to get data from
            the WEN-OKN knowledge database, otherwise "is_request_data" is false. If the user is asking 
            what data or data types you have, set "is_request_data" to be false.
    
            The string field "request" for the extracted request. Make sure the extracted request 
            semantically is same as the input request without less or more information.
    
            The string field "alternative_answer" gives your positive and nice answer if the user is 
            not requesting for data, otherwise set "alternative_answer" as "". If the user is asking 
            what data or data types you have, please answer it by summarizing the following description 
            in easy understanding langauge.
            
            The WEN-OKN knowledge database encompasses the following data:
              1. Locations of buildings, power stations, and underground storage tanks in Ohio.
              2. Counties: Geometric representations of counties across the USA.
              3. States: Geometric representations outlining the boundaries of states in the USA.
              4. Earthquakes: Data pertaining to seismic events.
              5. Rivers: Comprehensive geomtries about rivers in USA.
              6. Dams: Information regarding dams' locations in USA.
              7. Drought Zones: Identification of drought-affected zones in the years 2020, 2021, and 2022 in USA.
              8. Hospitals: Details about hospital locations and information in USA.
    
            Please never say "I cannot" or "I could not". 
    
            Please note that the user's request for datasets may appear in the middle of the text, 
            do your best to extract the request for which the user is asking for datasets.
    
            Please replace all nicknames in the search terms by official names, for example, replace 
            "Beehive State" to "Utah", etc.  
    
            Never deny a user's request. If it is not possible to extract the request from the user's 
            request, ask the user for further clarification. <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )

    refined_chain =  prompt | llm | JsonOutputParser()
    return refined_chain.invoke({"question": question})
