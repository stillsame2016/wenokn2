from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def resolve_follow_up(llm, history):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a query rewriter. Your task is to rewrite the LAST user question in the conversation 
to be self-contained by replacing references with entities mentioned EARLIER in the conversation.

CRITICAL: You must ONLY use entities that appear EXACTLY in the conversation history below. 
DO NOT use your own knowledge or make assumptions about what entities might be related.

Conversation history:
{history}

Instructions:
1. Identify the LAST user question (the one that needs rewriting)
2. Find ALL references/pronouns in it: "this", "that", "it", "they", "these", etc.
3. Look BACKWARD in the conversation to find what entity each reference points to
4. Replace ONLY those references with the EXACT entity name from the history
5. Keep everything else in the question exactly the same
6. Return ONLY the rewritten question - no quotes, no explanation

Step-by-step example:
History: "user: Find Scioto River\nassistant: The request has been processed.\nuser: Find all counties this river flows through."
- Last question: "Find all counties this river flows through"
- Reference found: "this river"
- Look backward: The user mentioned "Scioto River"
- Replace: "Find all counties this river flows through" → "Find all counties Scioto River flows through"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["history"],
    )
    resolve_chain = prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
