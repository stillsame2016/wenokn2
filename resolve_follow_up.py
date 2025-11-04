from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def resolve_follow_up(llm, history):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a query rewriter. Given a conversation history, your task is to rewrite the current question 
to be completely self-contained by replacing ALL references (pronouns, demonstratives like "this", "that", 
"these", "it", etc.) with their specific entities from the conversation history.

Conversation history:
{history}

Rules:
1. Replace ALL pronouns and references (this, that, it, they, etc.) with the actual entity names from history
2. Keep the same verb tense and structure as the original question
3. Return ONLY the rewritten question with no explanation, no quotes, no preamble
4. The rewritten question must be answerable without any conversation context

Examples:
- "Find all counties this river flows through" → "Find all counties Ohio River flows through"
- "What is its population?" → "What is California's population?"
- "Show me more details about it" → "Show me more details about the Eiffel Tower"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["history"],
    )
    resolve_chain = prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
