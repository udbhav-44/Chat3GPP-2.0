from typing_extensions import TypedDict
from typing import Annotated, List
from langchain_core.messages import BaseMessage
try:
    from langgraph.graph.message import add_messages
except ImportError:
    from langgraph.graph import add_messages
from Agents.LATS.Reflection import Node

class TreeState(TypedDict):
    # The full tree
    root: Node
    # The original input
    input: str
    # Conversation memory for prompt context
    messages: Annotated[List[BaseMessage], add_messages]
