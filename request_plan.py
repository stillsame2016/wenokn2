from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
# Implement the Request Planer
def get_request_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are the export of 
            WEN-OKN knowledge system which answer one question or return data for one entity type at 
            a time. For example, we can return dams or return earthquakes. You need to extract a list 
            of atomic requests from the user's question and also make each atomic request can be 
            executed independent to other atomic requests. 

            [Example 1]
            First find Scioto River and all dams on it. 
            
            This question requests to find a river entity and some dam entities satisfying some conditions. 
            So it can be divided into the following two atomic requests for two types of entities:
               Find Scioto River
               Find all dams on Scioto River
            
            [Example 2]
            First find Scioto River, then find all dams on this river. Also find all counties these dams locate. 
            
            This question requests to find a river entity and some dam entities and some county entities 
            satisfying some conditions. So it can be divided into the following three atomic requests for three
            entity types:
               Find Scioto River
               Find all dams on Scioto River
               Find all counties where dams on Scioto River are located

            [Example 3]
            Find all counties Scioto River flows through
            
            This question requests to find some county entities satisfying some conditions. So it is already an 
            atomic request.

            [Example 4]
            Find all dams on Scioto River

            This is already an atomic request because it requests to find dam entities satisfying some conditions.
            
            This question requests to find some county entities satisfying some conditions. So it is already an 
            atomic request.

            [Question]
            The following is the question from the user:
            {question}
            
            Think step by step and divide this question into atomic requests. Keep the atomic 
            request same as it is in the original question. Note that each atomic request should not contain any
            pronoun like "it" or "its" in order to handling each atomic request independently.
            
            Please return your answer in JSON with a list for those atomic request under a key "requests" 
            without preamble or explanation<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result
