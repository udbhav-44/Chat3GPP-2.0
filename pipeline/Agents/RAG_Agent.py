""""This file contains the code for the RAG Agent, which utilizes the RAG model to generate responses to the user query."""

import os
import json
import logging
from dotenv import load_dotenv
load_dotenv('.env')
logger = logging.getLogger(__name__)

from datetime import datetime

from langchain.globals import set_verbose
set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")

from Agents.LATS.OldfinTools import *

from LLMs import get_llm_for_role



def clean(text):
    return text[text.index('{'):text.rfind('}')+1]


def _search_and_generate_answer(query, meeting_id=""):
    try:
        result = search_and_generate.invoke({"query_str": query, "meeting_id": meeting_id})
    except Exception as exc:
        logger.exception("search_and_generate failed: %s", exc)
        return ""
    if isinstance(result, tuple) and len(result) == 2:
        return result[1] or ""
    return str(result)


def _simple_answer_from_response(resp):
    if isinstance(resp, dict):
        return resp.get("answer") or json.dumps(resp)
    return str(resp)


def ragAgent(query, state, model=None, provider=None):
    fin_context = ''''''
    has_uploads = has_user_uploads(get_current_user_id())
    
    if state == "report":
        sg_answer = _search_and_generate_answer(query)
        if sg_answer:
            fin_context += f"{sg_answer}\n"
        rag_result = ""
        if has_uploads:
            rag_result = retrieve_documents.invoke(query)
            fin_context += f"{rag_result} \n"
        sys_prompt =  '''
        Extract the Key Words, Jargons and Important Concepts from the information given below and make queries for further research:
        {
            "query_1": "...",
            "query_2": "...",
            "query_3": "..."
        }
        Following is the information to be used:
        \n
        '''
        rag_result_str = sg_answer or ""
        if rag_result:
            if type(rag_result) is list:
                for i in rag_result:
                    if isinstance(i, str):
                        rag_result_str += i
                    elif isinstance(i, dict):
                        url = i.get("url", "")
                        content = i.get("content", "")
                        rag_result_str += f"{url}+{content}"
            elif type(rag_result) is str:
                rag_result_str += rag_result
            else:
                logger.debug("Unexpected rag_result type: %s", type(rag_result))

        prompt = f"""Note: The Current Date and Time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. All your searches and responses must be with respect to this time frame""" + sys_prompt + rag_result_str
        llm = get_llm_for_role("complex", model=model, provider=provider, temperature=0.6, top_p=0.7)
        response = llm.invoke(f'''{prompt}''').content

        dic =  dict(json.loads(clean(response.split("```")[-2].split("json")[1])))
        for p in dic:
            if has_uploads:
                rag_resp = retrieve_documents.invoke(dic[p])
                fin_context += f'{rag_resp} \n'
            else:
                sg_followup = _search_and_generate_answer(dic[p])
                if sg_followup:
                    fin_context += f"{sg_followup}\n"



        return fin_context
        
    elif state == "concise":
        logger.info("Running concise RAG")
        sg_answer = _search_and_generate_answer(query)
        if not has_uploads:
            return sg_answer or "No matching 3GPP documents found."
        resp = simple_query_documents.invoke(query)
        user_answer = _simple_answer_from_response(resp)
        if sg_answer and user_answer:
            return f"{sg_answer}\n\nUser uploads context:\n{user_answer}"
        return sg_answer or user_answer
