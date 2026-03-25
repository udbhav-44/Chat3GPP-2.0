"""
This file contains the code for the Classifier Agent, 
which is used to classify the user query into different categories, based on the intent of the user.
"""


from dotenv import load_dotenv
from langchain_core.messages import SystemMessage

from LLMs import get_llm_for_role

load_dotenv('../../.env')
load_dotenv('.env')


def _invoke_classifier(prompt: str, model=None, provider=None) -> str:
    llm = get_llm_for_role(
        "classifier",
        model=model,
        provider=provider,
        temperature=0,
    )
    response = llm.invoke([SystemMessage(content=prompt)])
    return response.content if hasattr(response, "content") else str(response)

def classifierAgent(query, model=None, provider=None):
    
    prompt = f'''
    Instructions:
    1. For every input, think about the task, reason it out.
    2. But DO NOT WRITE THE REASONING IN THE OUTPUT. This is just for your thought process.
    3. Output should be a single word answer between 'simple' or 'complex'.

    A 'simple' query is a query in which the answer is concise and generating the answer does not require multiple agents.
    It might require multiple tools like web search, case law extractions etc, but the answer is a short answer or a paragraph or two.

    Questions like "Who am I?" or "What is the capital of France?", "What is 3GPP?", "What are the key features of LTE?", "What is the release date of 3GPP Release 17?", "List the 5G bands supported by 3GPP" etc.,
    require brief answers. Hence they are classified as 'simple'.

    A 'complex' query is query in which the answer should be a long report covering multiple aspects of the query, it will
    require multiple agents, complex orchestration. 

    Questions like "Analyze the impact of 5G on global markets and consumer behavior", "Compare and contrast the specifications of 3GPP Release 15 and Release 16",
    "What are the regulatory challenges in implementing 5G across different regions?", "Provide a detailed report on the evolution of 3GPP standards",
    "How does 3GPP address interoperability across different network vendors?" etc., are classified as 'complex'.

    Your task is to classify the given query as 'simple' or 'complex'. 
    
    Give a SINGLE word answer between 'simple' and 'complex'.

    Following is the query
    {query}
    '''
    return _invoke_classifier(prompt, model=model, provider=provider)

def classifierAgent_RAG(query, ragContext, model=None, provider=None):
    prompt = f'''
    You are a classifier which determines if a question asks for a detailed response or a concise response.
    
    A query is detailed if it expects a long response, in the form of a report, and expects in-depth analysis and reasoning, especially when discussing topics related to communication technologies, network performance, or 3GPP standardization.
    When a user expects a detailed response, they would mention key phrases like "in-depth analysis", "comprehensive report", "evaluate comprehensively", "multi-dimensional analysis", or "detailed analysis of 3GPP standards or network architectures".

    If the query has phrases like "analyze", "generate a report", "evaluate comprehensively", "detailed analysis", "comprehensive comparison", "impact assessment", "regulatory framework analysis", or other phrases that indicate the need for a thorough response, return 'detailed'.

    For example: 'Analyze the impact of 3GPP Release 16 on 5G adoption in emerging markets' is a detailed query because it requires multi-dimensional analysis covering technical, regulatory, and market aspects.

    A query which is not detailed is concise. Even if a query involves technical terms like "beamforming", "MIMO", or "5G core architecture", it can still be answered concisely if the response only requires brief explanations or definitions.
    For example: Questions like "What is 3GPP Release 15?", "Define beamforming in 5G", or "List key features of LTE" are concise queries because they can be answered with short, specific responses.

    Output should be a single word answer between 'concise' or 'detailed'.

    Following is the query:
    {query}
    '''
    cat_1 = _invoke_classifier(prompt, model=model, provider=provider).lower()

    if cat_1 == 'detailed':
        query2 = f'''''Does the following answer the query to its fullest extent? Evaluate on the following metrics, only return yes when all metrics are fulfilled:
        a. Completeness: Does the answer address all aspects of the query, particularly regarding 3GPP standards or communication systems?
        b. Detail: Does the answer provide the depth of detail required for the query? If the query asks for an in-depth analysis, does the answer meet that level of analysis?

        Query: {query}
        Answer: {ragContext}

        Answer only in 'yes' or 'no'
        '''
        
        cat_2 = _invoke_classifier(query2, model=model, provider=provider).lower()
        if cat_2 == 'yes':
            return 'simple'
        else:
            return 'complex' 
        
    else:
        return 'simple'
    
