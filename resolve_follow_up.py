from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

# def resolve_follow_up(llm, history):
#     # Programmatic check first
#     lines = history.strip().split('\n')
#     last_question = None
#     for line in reversed(lines):
#         if line.startswith('user:'):
#             last_question = line.replace('user:', '').strip()
#             break
    
#     if not last_question:
#         return history
    
#     # Check for reference words
#     reference_words = r'\b(this|that|it|its|they|them|their|these|those)\b'
#     has_reference = bool(re.search(reference_words, last_question, re.IGNORECASE))
    
#     if not has_reference:
#         return last_question
    
#     prompt = PromptTemplate(
#         template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
# You are a reference resolver. Rewrite the LAST user question by resolving pronouns.

# Conversation history:
# {history}

# CRITICAL RULES:
# 1. Each "user:" question produces a result (even if not shown)
# 2. When resolving "this/that/these/those", look at the IMMEDIATELY PREVIOUS user question to see what it asked for
# 3. The reference points to THE RESULT of that previous question, not to earlier entities
# 4. Chain the resolutions: if the previous question also had references, resolve those too
# 5. Do NOT add entities that aren't part of the reference chain

# STEP BY STEP:
# Step 1: Identify the last user question
# Step 2: Find ALL pronouns/references in it
# Step 3: For EACH reference:
#    - Look at the IMMEDIATELY PREVIOUS user question
#    - What did that question ask for? That's what the reference points to
#    - If that question ALSO had references, recursively resolve them
# Step 4: Replace the reference with the resolved phrase
# Step 5: Return ONLY the final rewritten question

# Example:
# History:
# user: Find the Scioto River
# assistant: Processed.
# user: Find the Ross county  
# assistant: Processed.
# user: Find all downstream counties of this river from this county
# assistant: Processed.
# user: Find populations of these counties
# assistant: Processed.
# user: Find all rivers that pass these counties

# Analysis:
# - Last question: "Find all rivers that pass these counties"
# - Reference: "these counties"
# - Previous question asked for: "populations of these counties"
# - But "populations" is just an attribute, the entities are still "these counties"
# - So look at what "these counties" referred to: "downstream counties of this river from this county"
# - Resolve "this river" → "Scioto River"
# - Resolve "this county" → "Ross county"
# - Final: "Find all rivers that pass the downstream counties of Scioto River from Ross county"

# IMPORTANT: Don't include entities mentioned earlier unless they're part of the reference resolution chain.

# <|eot_id|><|start_header_id|>assistant<|end_header_id|>
# """,
#         input_variables=["history"],
#     )
    
#     resolve_chain = prompt | llm | StrOutputParser()
#     return resolve_chain.invoke({"history": history})

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
You are a reference resolver. Rewrite the LAST user question by resolving ALL pronouns.

Conversation history:
{history}

CRITICAL RULE: References ALWAYS point to the MOST RECENT relevant entity.
- "these counties" = the counties from the IMMEDIATELY PREVIOUS user question
- "this river" = the MOST RECENTLY mentioned river
- Work backwards ONE question at a time

ALGORITHM:
1. Start with the last user question
2. Find a reference (this/that/these/those/it/them)
3. Go to the IMMEDIATELY PREVIOUS user question - what did it produce?
4. Replace the reference with what that question asked for
5. If that replacement ALSO contains references, recursively resolve them
6. Continue until no references remain

EXAMPLE TRACE:
Question 5: "Find all rivers that pass these counties"
→ "these counties" refers to previous question...

Question 4: "Find all neighboring counties of these counties"
→ So "these counties" in Q5 = "neighboring counties of these counties"
→ But Q4 also has "these counties", so resolve that...

Question 3: "Find all downstream counties of this river from this county"
→ So "these counties" in Q4 = "downstream counties of this river from this county"
→ But Q3 has "this river" and "this county", so resolve those...

Question 2: "Find the Ross county"
→ So "this county" = "Ross county"

Question 1: "Find the Scioto River"
→ So "this river" = "Scioto River"

RESOLVE STEP BY STEP:
- Q3 becomes: "downstream counties of Scioto River from Ross county"
- Q4 becomes: "neighboring counties of downstream counties of Scioto River from Ross county"
- Q5 becomes: "Find all rivers that pass the neighboring counties of downstream counties of Scioto River from Ross county"

Return ONLY the final rewritten question, no explanation.

<|eot_id|><|start_header_id|>assistant<|end_header_id|>
""",
        input_variables=["history"],
    )
    
    resolve_chain = prompt | llm | StrOutputParser()
    return resolve_chain.invoke({"history": history})
