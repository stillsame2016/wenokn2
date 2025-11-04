from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

def resolve_follow_up(llm, history):
  prompt = PromptTemplate(
        template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>
The input you receive will be the full conversation history in the format:
user: <question1>
assistant: <answer1>
user: <question2>
assistant: <answer2>
...
user: <current question>
Use this history to understand the context for follow-up questions.

The following is the conversation history:
{history}

Your task is to convert the current question into a self-contained independent question 
and return the converted question only without any explaination.
            <|eot_id|><|start_header_id|>assistant<|end_header_id|>
        """,
        input_variables=["history"],
    )

    resolve_chain =  prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
