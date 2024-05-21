from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
# Implement the Request Planer
def get_request_plan(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are the export of WEN-OKN knowledge 
            system which answer one question or return data for one entity type at a time. For example, we can return 
            dams or return earthquakes. You need to extract a list of atomic requests from the user's question and 
            also make each atomic request can be executed independent to other atomic requests. 
            
            [Example 1]
            First find Scioto River, then find all dams on this river. Also find all counties these 
            dams locate. 
            
            This question can be divided into the following atomic requests:
               Find Scioto River
               Find all dams on Scioto River
               Find all counties where dams on Scioto River are located

            [Question]
            The following is the question from the user:
            {question}
            
            Think step by step and divide this question into atomic requests. Keep the atomic 
            request same as it is in the original question.
            
            Please return your answer in JSON with a list for those atomic request under a key "request" 
            without preamble or explanation<|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_planer = prompt | llm | JsonOutputParser()
    result = question_planer.invoke({"question": question})
    return result
