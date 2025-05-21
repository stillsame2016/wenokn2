from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


#####################################################################
# Implement the Router
def check_report_request(llm, question):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> Check the question and return a JSON string in the
        following format:
        {{
            "create_report": true/false
        }}

        Set "create_report": true if the question implies intent to generate, produce, or create a report, summary, or document. Look for phrases like:
           - "generate a report"
           - "create a summary"
           - "compile a report"
           - "produce findings"
           - "document the results"

        Otherwise, set "create_report": false.

        Question to check: {question} 
        
        <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["question"],
    )
    question_router = prompt | llm | JsonOutputParser()
    result = question_router.invoke({"question": question})
    return result
