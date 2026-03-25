"""This file contains the code for the Planner Agent, which is responsible for generating the sub-agents that will be used to address the user query."""

import json
from pathlib import Path
from dotenv import load_dotenv
import time

import logging
from langchain.globals import set_verbose
import os

load_dotenv('.env')

set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")
logger = logging.getLogger(__name__)
WRITE_ARTIFACTS = os.getenv("WRITE_ARTIFACTS", "false").lower() in {"1", "true", "yes"}

from datetime import datetime
from LLMs import get_llm_for_role
from Agents.LATS.OldfinTools import has_user_uploads, get_current_user_id



def clean(text):
    return text[text.index('{'):text.rfind('}')+1]


def plannerAgent(query, model=None, provider=None, allow_web_tools=True):
    
    sys = f'''Note: The Current Date and Time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. All your searches and responses
        must be with respect to this time frame.'''
    has_uploads = has_user_uploads(get_current_user_id())
    uploads_notice = "yes" if has_uploads else "no"
    sys_prompt =  '''
    
    You are a task management assistant designed to break down tasks and manage task progress.

    While breaking down the main task into sub tasks, analyze from a multi-dimensional aspect, for instance interdependency
    For each of these domains like communication technologies, 3GPP standardization, network performance analysis, policy and regulation,
    etc, generate individual agents which do intensive research on these specific topics, leveraging the tools provided. 

    Each subtask should only cover one domain/aspect of the problem, and only one entity related to the problem, hence there
    can be many subtasks for a complex problem, and single or low number of tasks for a unidimensional problem.
    
    You can use multiple tools for each task. The research crew of agents must be extensive to ensure in-depth research.
    The main job in task breakdown is populating the JSON template below : 
    
    
    'json 

        {
        " main_task ": "..." , 
        " sub_tasks ": { 
        " task_1 ": {" content ": "..." , " agent ": "..." , "agent_role_description": "..." ," tools ": [...], " , " local_constraints ": [...] , " require_data ": [...]} ,
        " task_2 ": {" content ": "..." , " agent ": "..." , "agent_role_description": "..." ," tools ": [...], " local_constraints ": [...] , " require_data ": [...]}
        } 
        }
    '

    Before you design your task , you should understand what tools you have 
    , what each tool can do and cannot do. You must not design the subtask
    that do not have suitable tool to perform . Never design subtask that
    does not use any tool. Utilize specialized tools, like communication analysis tools,
    policy research tools, etc., over simple web searches, but if no specialized tool usage is possible,
    then use the scrape or search tools.

    Use search_and_generate as the PRIMARY tool for refutable knowledge directly from 3GPP documents.
    If user uploads are available, also use retrieve_documents/simple_query_documents for user-provided files.
    If user uploads are not available, do NOT use retrieve_documents/simple_query_documents/query_documents.
    Use web search only when specialized tools cannot answer the question.
     
    Based on user’s query , your main task is to gather valid information, create sub-tasks and synthesize agents which would execute these sub-tasks effectively. 
    
    You must first output the Chain of Thoughts ( COT ) . In the COT , you 
    need to explain how you break down the main task into sub - tasks and
    justify why each subtask can be completed by a singular agent which you synthesize. The sub - tasks
    need to be broken down to a very low granularity , hence it ’ s possible
    that some sub - tasks will depend on the execution results of previous
    tasks . You also need to specify which sub - tasks require the execution
    results of previous tasks . When writing about each sub - task , you must
    also write out its respective local constraints . Finally , you write
    the global constraint of the main task. While applying Chain of thought, think
    about related domains, topics and issues in order to take an interdisciplinary
    and holistic approach. 
    
    Try to maximize the number of independent tasks so that we can run them parallelly but
    where dependence from previous tasks is necessary, do add dependence.
    
    Before filling in the template , you must first understand the user ’ s 
    request , carefully analyzing the tasks contained within it . Once you
    have a clear understanding of the tasks , you determine the sequence in
    which each task should be executed . Following this sequence , you
    rewrite the tasks into complete descriptions , taking into account the
    dependencies between them.
    
    In the JSON template you will be filling , " main_task " is your main 
    task , which is gather valid information based on user ’ s
    query . " sub_task " is the sub - tasks that you would like to break down
    the task into . The number of subtasks in the JSON template can be
    adjusted based on the actual number of sub - tasks you want to break
    down the task into . The break down process of the sub - tasks must be
    simple with low granularity . There is no limit to the number of
    subtasks. 

    Each sub - tasks consist of either one or multiple step . It contains 6
    information to be filled in , which are " content " , " agent " , "agent_role_description", "tools"  , " require_data " and " data ".
    
    " require_data " is a list of previous sub - tasks which their information 
    is required by the current sub - task . Some sub - tasks require the
    information of previous sub - task . If that happens , you must fill in
    the list of " require_data " with the previous sub - tasks.

    Note: require_data should contain a list of task names like "task_1", "task_2"
    etc, and nothing else. Ensure that the strings match with the task names strictly. 
    
    " content " is the description of the subtask , formatted as string . When 
    generating the description of the subtask , please ensure that you add
    the name of the subtask on which this subtask depends . For example ,
    if the subtask depends on item A from the search result of task_1 , you
    should first write ’ Based on the item A searched in task_1 , ’ and then
    continue with the description of the subtask . It is important to
    indicate the names of the dependent subtasks .
    
     " agent " is the agent required for each step of execution. For each subtask there must
    only be one agent. Please use the original name of the agent synthesized.
    This list cannot be empty. If you could not think of any agent to
    perform this sub-task, please do not write this sub-task.
    Examples of agents might include: Communication Systems Researcher, 3GPP Standards Specialist, 
    Regulatory Analyst, Policy Advisor, Financial Analyst, Market Researcher, Technical Documentation Specialist, etc. Note that this is not an exhaustive list and you can 
    make other agents on the same lines.
    
    "agent_role_description" is the detailed job role  description of the agent and
    it's specializations which are required to solve the specific task. This is a 
    detailed string which describes what the agent is supposed to do and what output
    and specialization is expected from that agent.
    
    " tools " is the list of tools required for each step of execution . 
    Please use the original name of the tool without " functions ." in front. 
    This list cannot be empty . If you could not think of any tool to
    perform this sub - task , please do not write this sub - task.
    DO NOT add function arguments or parameters in front of the tool names. 
    The tool names must be the same as the name of the functions provided 
    and nothing else.

    After determining your subtasks , you must first identify the local 
    constraints for each sub - task , then the global constraints . Local
    constraints are constraints that needed to be considered in only in
    each specific sub - task.
    Please write the local constraints of each sub - task in its 
    corresponding " local_constraints " Local constraints of each sub - 
    task must be unique .
    When writing " local_constraints " , please write it as specific as 
    possible , as you should assume the agents of each task have no
    knowledge of the user ’ s query . You should also be aware local
    constraints filters the items individually , and some constraints can
    only be satisfied by multiple items .

    Never design subtask that uses tools which are beyond the functionality
    of LLMs or the tools defined. Also don't mention a tool not present in
    the tools list provided.

    You must output the JSON at the end .
    The Query is Given as follows:
    '''

    prompt = sys + sys_prompt + f"{query}"


    def load_json(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)  # Parse JSON into a Python dictionary or list
                return data
        except FileNotFoundError:
            logger.error("The file '%s' was not found.", file_path)
            return None
        except json.JSONDecodeError as e:
            logger.exception("Error decoding JSON")
            return None

    # Access specific information from the JSON
    def get_value_from_json(data, key):
        try:
            value = data.get(key)  # Fetch value associated with the key
            return value
        except AttributeError:
            logger.error("JSON data is not a dictionary.")
            return None

    # Resolve tools file relative to this module so it works regardless of CWD
    TOOLS_PATH = Path(__file__).resolve().parents[1] / "Tools" / "info.json"
    json_data = load_json(TOOLS_PATH)
    if not json_data:
        raise RuntimeError(f"Could not load tools metadata from {TOOLS_PATH}")
    if not has_uploads:
        blocked = {"simple_query_documents", "retrieve_documents", "query_documents"}
        json_data = [tool for tool in json_data or [] if tool.get("name") not in blocked]
    if not allow_web_tools:
        blocked = {"web_search", "web_scrape", "web_search_simple"}
        json_data = [tool for tool in json_data or [] if tool.get("name") not in blocked]


    tools_prompt = f'''
    The information about tools is encoded in a list of dictionaries, each dictionary having four keys: with 4 keys: first key being the 'name' where you enter the name of the function, second being 'docstring', where you fill the docstring of the function and third being 'parameters', fourth being output where you mention the output and output type. 
    NOTE that you can only use these tools and not anything apart from these. The names of the following tools are the only names valid for any of the tools. 
    The tools available to us are as follows:

    {json_data}

    '''
    prompt = prompt + tools_prompt + f"\nUser uploads available: {uploads_notice}\n"

    llm = get_llm_for_role("complex", model=model, provider=provider, temperature=0.6, top_p=0.7)
    response = llm.invoke(f'''{prompt}''').content
    dic =  json.loads(clean(response.split("```")[-2].split("json")[1]))


    if WRITE_ARTIFACTS:
        agent_dir = Path(__file__).resolve().parent
        with open(agent_dir / 'plan.txt', 'w') as f:
            f.write(response)

        with open(agent_dir / 'plan.json', 'w') as f:
            json.dump(dic, f)

    return dic



def plannerAgent_rag(query, ragContent, model=None, provider=None, allow_web_tools=True):
    
    sys = f'''Note: The Current Date and Time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. All your searches and responses
        must be with respect to this time frame.'''
    has_uploads = has_user_uploads(get_current_user_id())
    uploads_notice = "yes" if has_uploads else "no"
    sys_prompt =  '''
    
    You are a task planning assistant designed to break down tasks and manage task progress based on a prompt and context from a document.

    While breaking down the main task into sub tasks, analyze from a multi-dimensional aspect, for instance interdependency
    between multiple domains like communication systems, mobile technologies, 3GPP standards, regulatory aspects, market strategies, and consumer adoption.
    Large Scale considerations v/s Small Scale considerations, Long Term Considerations v/s Short Term Considerations etc. 
    For each of these domains like communication technologies, 3GPP standardization, network performance analysis, policy and regulation, finance, market 
    strategy etc, generate individual agents which do intensive research on these specific topics, leveraging the tools provided.
    Use search_and_generate as the PRIMARY tool for refutable knowledge directly from 3GPP documents.
    If user uploads are available, also use retrieve_documents/simple_query_documents for user-provided files.
    If user uploads are not available, do NOT use retrieve_documents/simple_query_documents.
    Use web search only when specialized tools cannot answer the question.
    The agents should focus HEAVILY ON extracting numbers from the context. Extract AS MANY NUMBERS as possible. Also, the source provided to you should be mentioned EXPLICITLY.


    The task divison should be very specific to the context. Following is the context:

    =======================================================
    {ragContent}
    =======================================================

    User uploads available: ''' + uploads_notice + '''

    
    
    You can use multiple tools for each task. The research crew of agents must be extensive to ensure in depth research.
    The main job in task breakdown is populating the JSON template below : 
    
    'json 

        {
        " main_task ": "..." , 
        " sub_tasks ": { 
        " task_1 ": {" content ": "..." , " agent ": "..." , "agent_role_description": "..." ," tools ": [...], " , " local_constraints ": [...] , " require_data ": [...]} ,
        " task_2 ": {" content ": "..." , " agent ": "..." , "agent_role_description": "..." ," tools ": [...], " local_constraints ": [...] , " require_data ": [...]}
        } 
        }
    '

    Before you design your task, you should understand what tools you have,
    what each tool can do and cannot do. You must not design the subtask
    that does not have a suitable tool to perform. Never design a subtask that
    does not use any tool. Utilize specialized tools, like communication analysis tools,
    policy research tools, etc., over simple web searches, but if no specialized tool usage is possible,
    then use the scrape or search tools.
     
    Try to minimize the number of tasks, but make at least 3 tasks.
    
    In the JSON template you will be filling , " main_task " is your main 
    task , which is gather valid information based on user ’ s
    query . " sub_task " is the sub - tasks that you would like to break down
    the task into . 

    Each sub - tasks consist of either one or multiple step . It contains 6
    information to be filled in , which are " content " , " agent " , "agent_role_description", "tools"  , " require_data " and " data ".
    
    "require_data " is a list of previous sub - tasks which their information 
    is required by the current sub - task . Some sub - tasks require the
    information of previous sub - task . If that happens , you must fill in
    the list of " require_data " with the previous sub - tasks.

    Note: require_data should contain a list of task names like "task_1", "task_2"
    etc, and nothing else. Ensure that the strings match with the task names strictly. 
    
    " content " is the description of the subtask , formatted as string.
    
    " agent " is the agent required for each step of execution . For each subtask there must
    only be one agent.Please use the original name of the agent synthesized.
    . This list cannot be empty . If you could not think of any agent to
    perform this sub - task , please do not write this sub - task.
    
    "agent_role_description" is the detailed job role  description of the agent and
    it's specializations which are required to solve the specific task. 
    
    " tools " is the list of tools required for each step of execution . 
    Please use the original name of the tool without " functions ." in front. 
    
    DO NOT add function arguments or parameters in front of the tool names. 
    The tool names must be the same as the name of the functions provided 
    and nothing else.

    After determining your subtasks , you must first identify the local 
    constraints for each sub - task. Local
    constraints are constraints that needed to be considered in only in
    each specific sub - task.
    Please write the local constraints of each sub - task in its 
    corresponding " local_constraints " Local constraints of each sub - 
    task must be unique .
    When writing " local_constraints " ,

    Never design subtask that uses tools which are beyond the functionality
    of LLMs or the tools defined. Also don't mention a tool not present in
    the tools list provided.

    You must output the JSON at the end .
    The Query is Given as follows:
    '''

    prompt = sys + sys_prompt + f"{query}"


    def load_json(file_path):
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)  # Parse JSON into a Python dictionary or list
                return data
        except FileNotFoundError:
            logger.error("The file '%s' was not found.", file_path)
            return None
        except json.JSONDecodeError as e:
            logger.exception("Error decoding JSON")
            return None

    # Access specific information from the JSON
    def get_value_from_json(data, key):
        try:
            value = data.get(key)  # Fetch value associated with the key
            return value
        except AttributeError:
            logger.error("JSON data is not a dictionary.")
            return None

    # Example usage
    file_path = 'Tools/info.json'  # Replace with your JSON file path
    json_data = load_json(file_path)
    if not allow_web_tools:
        blocked = {"web_search", "web_scrape", "web_search_simple"}
        json_data = [tool for tool in json_data if tool.get("name") not in blocked]


    tools_prompt = f'''
    The information about tools is encoded in a list of dictionaries, each dictionary having four keys: with 4 keys: first key being the 'name' where you enter the name of the function, second being 'docstring', where you fill the docstring of the function and third being 'parameters', fourth being output where you mention the output and output type. 
    NOTE that you can only use these tools and not anything apart from these. The names of the following tools are the only names valid for any of the tools. 
    The tools available to us are as follows:

    {json_data}

    '''
    prompt = prompt + tools_prompt

    llm = get_llm_for_role("complex", model=model, provider=provider, temperature=0.6, top_p=0.7)
    response = llm.invoke(f'''{prompt}''').content
    dic =  json.loads(clean(response.split("```")[-2].split("json")[1]))


    if WRITE_ARTIFACTS:
        with open('./Agents/plan.txt', 'w') as f:
            f.write(response)

        with open('./Agents/plan.json', 'w') as f:
            json.dump(dic, f)

    return dic




if __name__ == "__main__":
    start = time.time()
    query = 'Analyze the impact of US-China trade wars on multiple financial assets'
    out = plannerAgent(query)
    logger.info("Planning complete")
    logger.info("Time for planning: %.2fs", time.time() - start)
