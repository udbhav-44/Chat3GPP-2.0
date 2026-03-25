""" 
This file contains the Agent class which is used to create an agent object for each task in the LATS pipeline.
functions:
    - __init__ : initializes the agent object with the task number, name, role, constraints, task, dependencies, tools_list, state
    - genContext_andRunLATS : generates the context for the agent based on the dependencies and runs the LATS pipeline for the agent
    - Agent : class to create an agent object for each task in the LATS pipeline
"""
import google.generativeai as genai
from Agents.LATS.OldfinTools import *
from datetime import datetime
import os
from langchain.globals import set_verbose
set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")
import logging
from Agents.LATS.Solve_subquery import SolveSubQuery
logger = logging.getLogger(__name__)

class Agent:
    def __init__(
        self,
        number,
        name,
        role,
        constraints,
        task,
        dependencies,
        tools_list,
        state,
        thread_id=None,
        model=None,
        provider=None,
        allow_web_tools=True,
    ):
        self.taskNumber = number
        self.name = name
        self.role = role
        self.constraints = constraints
        self.dependencies = dependencies
        self.context = ''
        self.task = task
        self.state = state
        self.thread_id = thread_id
        self.model = model
        self.provider = provider
        self.allow_web_tools = allow_web_tools

        tl_lis = []
        has_uploads = has_user_uploads(get_current_user_id())

        if len(tools_list) < 1 and self.allow_web_tools:
            tools_list.append('web_search')

        for function_name in tools_list:
            if '(' in function_name:
                function_name = function_name.split('(')[0]
            if not has_uploads and function_name in {
                "simple_query_documents",
                "retrieve_documents",
                "query_documents",
            }:
                continue
            tl_lis.append(globals()[function_name])
        self.tools_list = tl_lis[:]
        
        if self.state == 'RAG' and has_uploads:
            self.tools_list.append(retrieve_documents)

        self.PREFIX_TEMPLATE = f"""You are a {self.name}, with the following role : {self.role}."""
        self.CONSTRAINT_TEMPLATE = f"the constraint is {self.constraints}. "

        self.func_docs = ''''''

        for func in self.tools_list:
            self.func_docs+=f'''{func.name}: {func.description}\n'''

        
    def genContext_andRunLATS(self, response_dict):
        for task in self.dependencies:
            if task in response_dict:
                self.context += response_dict[task]
            else:
                logger.warning("%s not executed yet, expected before %s", task, self.taskNumber)

        ROLE_TEMPLATE = f"""

            Note: The Current Date and Time is {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}. All your searches and responses
            must be with respect to this time frame.

            IMPORTANT: DO NOT REMOVE ANY SOURCE LINKS. FORMAT THEM ACCORDING TO MARKDOWN. 
            Cite all the sources, website links, and data sources

            Make your response on the basis of the history: 
            {self.context} 

            and the specific subtask for you {self.task}

            Based on your role, research and give us a comprehensive response analyzing various metrics. Try to stick to your role. 
            Try to substantiate your answers with as much technical details, numerical analysis,formulaes, concepts, regulatory considerations, and historical context as possible wherever required.
            Provide numbers, concepts and explicitly researched facts in your response in order to back your claims. You may also provide tables of 
            relevant information.
            Research, analyze, and report from a multi-dimensional aspect, focusing on interdependencies between communication systems, 
            3GPP standards, market dynamics, regulatory frameworks, and technology adoption trends. Consider Large Scale considerations v/s Small Scale considerations, 
            Long Term Considerations v/s Short Term Considerations, etc.
            You have access to the following tools:

            {self.func_docs}

            Use the following format for reasoning:
            - Thought: Describe what you're thinking.
            - Action: Choose a tool from the pool of tools.
            - Action Input: Provide the input for the tool. Ensure that the input provided matches with the parameters of the tool, and the datatypes are same.
            - Observation: Record the tool's result. In this observation, give a detailed explanation and reasoning of your response, backed by technical details, facts and numbers wherever required.
            Ensure that the Observation, that is the Final response is not short and concise, but detailed report with all the facts and figures well substantiated. 
            Try to substantiate your answers with as much technical details, comparisons, regulatory analysis, and historical context as possible wherever required.
            MAKE YOUR OUTPUTS EXTREMELY DETAILED AND WELL REASONED AND DO NOT OMIT ANY IMPORTANT FACTS WHICH ARE RESEARCHED BY THE TOOLS.

            IMPORTANT: Cite all the sources, website links, and data sources at the location where information is mentioned. 
            All links must be functional and correspond to the data. Cite the links at the location of the data, and at the end
            of the report generated. This is EXTREMELY IMPORTANT. THESE LINKS SHOULD BE CLICKABLE.
        """
        PROMPT_TEMPLATE = self.PREFIX_TEMPLATE + self.CONSTRAINT_TEMPLATE + ROLE_TEMPLATE
        
        checkpoint_ns = f"{self.name}-{self.taskNumber}"
        response = SolveSubQuery(
            PROMPT_TEMPLATE,
            self.tools_list,
            thread_id=self.thread_id,
            checkpoint_ns=checkpoint_ns,
            model=self.model,
            provider=self.provider,
        )

        return response


        

        
