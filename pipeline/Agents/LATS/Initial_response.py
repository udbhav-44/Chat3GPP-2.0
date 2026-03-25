""" 
This file contains the code to generate the initial response for the LATS agent. 
functions:
    - custom_generate_initial_response: This function generates the initial response for the LATS agent.
"""
from langchain_core.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from Agents.LATS.TreeState import TreeState
from Agents.LATS.Reflection import reflection_chain, Node
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langgraph.prebuilt import ToolNode
from Agents.LATS.OldfinTools import *
from dotenv import load_dotenv
load_dotenv('.env')

from LLMs import get_llm

prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                Generate a detailed response backed by numbers and sources to the user question below
                1. Use 'search_and_generate' as the primary source of refutable knowledge directly from 3GPP documents.
                2. If 'retrieve_documents' is present, use it only for user-uploaded documents and cite document/page references.
                3. If 'retrieve_documents' yields sufficient user context, you may skip additional user-doc tools.
                4. Use web search tools only if specialized tools cannot answer the query.
                5. Cite the sources and the exact webpage link next to the relevant information in each response.
                """,
            ),
            MessagesPlaceholder(variable_name="messages", optional=True),
            ("user", "{input}"),
        ]
    )

# Define the node we will add to the graph
def custom_generate_initial_response(tools, model=None, provider=None):
    """ 
    Generate the initial response for the LATS agent.
    Args:
        tools: The tools available to the agent.
    Returns:
        function: The function to generate the initial response.
    """
    tool_node = ToolNode(tools=tools)
    def generate_initial_response(state: TreeState) -> dict:
        llm = get_llm(model=model, provider=provider, role="lats")
        initial_answer_chain = prompt_template | llm.bind_tools(tools=tools).with_config(
            run_name="GenerateInitialCandidate"
        )

        parser = JsonOutputToolsParser(return_id=True)

        #Generate the initial candidate response.
        res = initial_answer_chain.invoke(
            {"input": state["input"], "messages": state.get("messages")}
        )
        parsed = parser.invoke(res)
        tool_responses = [
            tool_node.invoke(
                {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {"name": r["type"], "args": r["args"], "id": r["id"]}
                            ],
                        )
                    ]
                }

            )
            for r in parsed
        ]
        output_messages = [res] + [tr["messages"][0] for tr in tool_responses]
        reflection = reflection_chain.invoke(
            {
                "input": state["input"],
                "candidate": output_messages,
                "_model": model,
                "_provider": provider,
            }
        )
        root = Node(output_messages, reflection=reflection)
        return {
            **state,
            "root": root,
        }
    return generate_initial_response
