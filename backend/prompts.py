"""Default system prompts for the LLM Council Plus."""

STAGE1_PROMPT_DEFAULT = """You are a helpful AI assistant.
{search_context_block}
Question: {user_query}"""

STAGE1_SEARCH_CONTEXT_TEMPLATE = """You have access to the following real-time web search results.
You MUST use this information to answer the question, even if it contradicts your internal knowledge cutoff.
Do not say "I cannot access real-time information" or "My knowledge is limited to..." because you have the search results right here.

Search Results:
{search_context}
"""

STAGE2_PROMPT_DEFAULT = """You are evaluating different responses to the following question:

Question: {user_query}

{search_context_block}
Here are the responses from different models (anonymized):

{responses_text}

Your task:
1. First, evaluate each response individually. For each response, explain what it does well and what it does poorly.
2. Then, at the very end of your response, provide a final ranking.

IMPORTANT: Your final ranking MUST be formatted EXACTLY as follows:
- Start with the line "FINAL RANKING:" (all caps, with colon)
- Then list the responses from best to worst as a numbered list
- Each line should be: number, period, space, then ONLY the response label (e.g., "1. Response A")
- Do not add any other text or explanations in the ranking section

Example of the correct format for your ENTIRE response:

Response A provides good detail on X but misses Y...
Response B is accurate but lacks depth on Z...

FINAL RANKING:
1. Response A
2. Response B

Rank ONLY the responses listed above. Do not invent labels that were not provided.

Now provide your evaluation and ranking:"""

STAGE3_PROMPT_DEFAULT = """You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses.

Original Question: {user_query}

{search_context_block}
STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

TITLE_PROMPT_DEFAULT = """Generate a very short title (3-5 words maximum) that summarizes the following question.
The title should be concise and descriptive. Do not use quotes or punctuation in the title.

Question: {user_query}

Title:"""

QUERY_PROMPT_DEFAULT = """Extract the core search terms from the following question to use in a web search engine.
Return ONLY the search query, with no quotes, punctuation, or conversational text.

Question: {user_query}

Search Query:"""
