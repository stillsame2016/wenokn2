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
    # Programmatic check first
    lines = history.strip().split('\n')
    last_question = None
    for line in reversed(lines):
        if line.startswith('user:'):
            last_question = line.replace('user:', '').strip()
            break
    
    if not last_question:
        return history
    
    # Check for reference words
    reference_words = r'\b(this|that|it|its|they|them|their|these|those)\b'
    has_reference = bool(re.search(reference_words, last_question, re.IGNORECASE))
    
    if not has_reference:
        return last_question
    
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a reference resolver. Rewrite the LAST user question by resolving pronouns.

Conversation history:
{history}

CRITICAL RULES:
1. Each "user:" question produces a result (even if not shown)
2. When resolving "this/that/these/those", look at the IMMEDIATELY PREVIOUS user question to see what it asked for
3. The reference points to THE RESULT of that previous question, not to earlier entities
4. Chain the resolutions: if the previous question also had references, resolve those too
5. Do NOT add entities that aren't part of the reference chain

STEP BY STEP:
Step 1: Identify the last user question
Step 2: Find ALL pronouns/references in it
Step 3: For EACH reference:
   - Look at the IMMEDIATELY PREVIOUS user question
   - What did that question ask for? That's what the reference points to
   - If that question ALSO had references, recursively resolve them
Step 4: Replace the reference with the resolved phrase
Step 5: Return ONLY the final rewritten question

Example:
History:
user: Find the Scioto River
assistant: Processed.
user: Find the Ross county  
assistant: Processed.
user: Find all downstream counties of this river from this county
assistant: Processed.
user: Find populations of these counties
assistant: Processed.
user: Find all rivers that pass these counties

Analysis:
- Last question: "Find all rivers that pass these counties"
- Reference: "these counties"
- Previous question asked for: "populations of these counties"
- But "populations" is just an attribute, the entities are still "these counties"
- So look at what "these counties" referred to: "downstream counties of this river from this county"
- Resolve "this river" → "Scioto River"
- Resolve "this county" → "Ross county"
- Final: "Find all rivers that pass the downstream counties of Scioto River from Ross county"

IMPORTANT: Don't include entities mentioned earlier unless they're part of the reference resolution chain.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["history"],
    )
    
    resolve_chain = prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
