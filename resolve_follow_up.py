from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# def resolve_follow_up(llm, history):
#     prompt = PromptTemplate(
#         template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a query rewriter. Your task is to rewrite the LAST user question in the conversation 
# to be self-contained by replacing references with entities mentioned EARLIER in the conversation.

# CRITICAL: You must ONLY use entities that appear EXACTLY in the conversation history below. 
# DO NOT use your own knowledge or make assumptions about what entities might be related.

# Conversation history:
# {history}

# Instructions:
# 1. Identify the LAST user question (the one that needs rewriting)
# 2. Find ALL references/pronouns in it: "this", "that", "it", "they", "these", etc.
# 3. Look BACKWARD in the conversation to find what entity each reference points to
# 4. Replace ONLY those references with the EXACT entity name from the history
# 5. Keep everything else in the question exactly the same
# 6. Return ONLY the rewritten question - no quotes, no explanation

# Step-by-step example:
# History: "user: Find Scioto River\nassistant: The request has been processed.\nuser: Find all counties this river flows through."
# - Last question: "Find all counties this river flows through"
# - Reference found: "this river"
# - Look backward: The user mentioned "Scioto River"
# - Replace: "Find all counties this river flows through" → "Find all counties Scioto River flows through"

# [ Example ]
# History: "user: Find Washington County in Oregon\nassistant: Your request has been processed.\nuser: Find Washington County"
# - Last question: "Find Washington County"
# - No references/pronouns in the last question
# - Return "Find Washington County"
# Note: The first question is finding a county with the nams 'Washington' in Oregon. 
# The current question is finding all counties with the name 'Washington'. 
# The two questions doesn't have any relations.

# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """,
#         input_variables=["history"],
#     )
#     resolve_chain = prompt | llm | StrOutputParser()
#     return resolve_chain.invoke({"history": history})


import re

def resolve_follow_up(llm, history):
    # Extract last user question
    lines = history.strip().split('\n')
    last_question = None
    for line in reversed(lines):
        if line.startswith('user:'):
            last_question = line.replace('user:', '').strip()
            break
    
    if not last_question:
        return history
    
    # Programmatically check for reference words
    reference_words = r'\b(this|that|it|its|they|them|their|these|those|the same|such)\b'
    has_reference = bool(re.search(reference_words, last_question, re.IGNORECASE))
    
    # If no references found, return unchanged
    if not has_reference:
        return last_question
    
    # Only invoke LLM if references detected
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
Rewrite the LAST user question by replacing references/pronouns with entities from earlier in the conversation.

Conversation history:
{history}

Rules:
- Find what "this", "that", "it", "they", etc. refer to in the conversation
- Replace them with the actual entity names
- Return ONLY the rewritten question

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["history"],
    )
    resolve_chain = prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
