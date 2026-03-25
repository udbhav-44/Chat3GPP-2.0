from Agents.LATS.Solve_subquery import SolveSubQuery

import json
import logging
import os
from dotenv import load_dotenv


from LLMs import run_conversation_complex

load_dotenv('.env')
logger = logging.getLogger(__name__)
WRITE_ARTIFACTS = os.getenv("WRITE_ARTIFACTS", "false").lower() in {"1", "true", "yes"}

def drafterAgentSimplified(text, query, model=None, provider=None):
    system_prompt = f'''
    Your ultimate task is to give a comprehensive answer to the query:{query}
    Judge the length of the response on the basis of the query and generate the response accordingly.
    '''
    user_prompt = f'''
    Following is the content:
    {text}
    
    '''
    prompt = f'''{system_prompt}\n\n {user_prompt}'''

    response = run_conversation_complex(f'''{prompt}''', model=model, provider=provider)
    
    return response

def conciseAns_vanilla_LATS(query, tools_list, model=None, provider=None):
    logger.info("Running conciseAns_vanilla_LATS")
    CombinedResearch = [SolveSubQuery(query, tools=tools_list, model=model, provider=provider)]
    CombinedResearch_json = json.dumps(CombinedResearch,indent=2)
    fin_resp = drafterAgentSimplified(CombinedResearch_json, query, model=model, provider=provider)
    if WRITE_ARTIFACTS:
        with open("conciseResponse_LATS.md", "w") as f1:
            f1.write(fin_resp)
    logger.info("Completed conciseAns_vanilla_LATS")
    return fin_resp
