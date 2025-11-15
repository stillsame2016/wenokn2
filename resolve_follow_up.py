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


def resolve_follow_up(llm, history):
    prompt = PromptTemplate(
        template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a query rewriter. Your task is to rewrite the LAST user question ONLY IF it contains 
references/pronouns that need to be resolved.

Conversation history:
{history}

Instructions:
1. Identify the LAST user question
2. Check if it contains ANY references/pronouns: "this", "that", "it", "they", "these", "them", "those", etc.
3. If NO references/pronouns found → Return the question UNCHANGED
4. If references/pronouns found:
   - Look BACKWARD in the conversation to find what entity each reference points to
   - Replace ONLY those references with the EXACT entity name from the history
   - Keep everything else exactly the same
5. Return ONLY the rewritten question - no quotes, no explanation

CRITICAL RULE: If the last question contains NO pronouns or references, return it EXACTLY as written.

Examples:

Example 1 (HAS reference):
History: "user: Find Scioto River\nassistant: The request has been processed.\nuser: Find all counties this river flows through."
- Last question: "Find all counties this river flows through"
- Reference found: "this river"
- Look backward: Found "Scioto River"
- Output: "Find all counties Scioto River flows through"

Example 2 (NO reference):
History: "user: Find Washington County in Oregon\nassistant: Your request has been processed.\nuser: Find Washington County"
- Last question: "Find Washington County"
- Check for references: NO pronouns or references like "this", "that", "it", etc.
- Output: "Find Washington County"

Example 3 (NO reference - different search):
History: "user: Find rivers in California\nassistant: Your request has been processed.\nuser: Find rivers in Oregon"
- Last question: "Find rivers in Oregon"
- Check for references: NO pronouns or references
- These are two independent queries about different states
- Output: "Find rivers in Oregon"

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["history"],
    )
    resolve_chain = prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
