"""
This file is used to generate the graph for the LATS agent. 
The graph is generated using the StateGraph class from langgraph.
"""
import os
import logging
import atexit
from typing import Literal
from Agents.LATS.Initial_response import custom_generate_initial_response
from Agents.LATS.TreeState import TreeState
from Agents.LATS.generate_candiates import custom_expand
from Agents.LATS.CheckpointSerde import LatsJsonPlusSerializer
from langgraph.graph import END, StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from Agents.LATS.OldfinTools import *
from dotenv import load_dotenv
from langgraph.checkpoint.base import maybe_add_typed_methods

try:
    from langgraph.checkpoint.postgres import PostgresSaver
except ImportError:
    PostgresSaver = None

load_dotenv('.env')
logger = logging.getLogger(__name__)

def _lats_checkpoint_enabled() -> bool:
    return os.getenv("LATS_CHECKPOINTS_ENABLED", "false").lower() in {"1", "true", "yes"}

_POSTGRES_SAVER = None
_POSTGRES_SAVER_CTX = None

def _close_postgres_saver():
    global _POSTGRES_SAVER_CTX, _POSTGRES_SAVER
    if _POSTGRES_SAVER_CTX is None:
        return
    try:
        _POSTGRES_SAVER_CTX.__exit__(None, None, None)
    except Exception:
        logger.exception("Failed to close Postgres checkpointer")
    _POSTGRES_SAVER_CTX = None
    _POSTGRES_SAVER = None

def _open_postgres_saver(conn_str: str):
    global _POSTGRES_SAVER_CTX, _POSTGRES_SAVER
    if _POSTGRES_SAVER is not None:
        return _POSTGRES_SAVER
    _POSTGRES_SAVER_CTX = PostgresSaver.from_conn_string(conn_str)
    _POSTGRES_SAVER = _POSTGRES_SAVER_CTX.__enter__()
    lats_serde = LatsJsonPlusSerializer()
    _POSTGRES_SAVER.serde = maybe_add_typed_methods(lats_serde)
    _POSTGRES_SAVER.jsonplus_serde = lats_serde
    atexit.register(_close_postgres_saver)
    return _POSTGRES_SAVER

def get_checkpoint_conn_str():
    url = os.getenv("LANGGRAPH_CHECKPOINT_URL") or os.getenv("CHECKPOINT_DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "udbhav")
    password = os.getenv("POSTGRES_PASSWORD", "login123")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def build_checkpointer():
    if not _lats_checkpoint_enabled():
        logger.info("LATS checkpointing disabled; set LATS_CHECKPOINTS_ENABLED=true to enable.")
        return None
    if PostgresSaver is None:
        logger.info("Postgres checkpointer unavailable; disabling LATS checkpointing.")
        return None
    conn_str = get_checkpoint_conn_str()
    try:
        saver = _open_postgres_saver(conn_str)
        if hasattr(saver, "setup"):
            saver.setup()
        return saver
    except Exception as exc:
        logger.warning("Postgres checkpointer unavailable (%s); disabling LATS checkpointing.", exc)
        return None

CHECKPOINTER = build_checkpointer()

def should_loop(state: TreeState):
    """
    Determine whether to continue the tree search.
    Args:
        state: The current state of the tree search.
    Returns:
        Literal["expand", "finish"]: Whether to continue the tree search.
    """
    root = state["root"]
    if root.is_solved:
        return END
    if root.height > 3:
        return END
    return "expand"

def generateGraph_forLATS(tools, model=None, provider=None):
    """ Generate the graph for the LATS agent.
    Args:
        tools: The tools available to the agent.
    Returns:
        StateGraph: The graph for the LATS agent.
    """
    builder = StateGraph(TreeState)
    builder.add_node(
        "start",
        custom_generate_initial_response(tools, model=model, provider=provider),
    )
    builder.add_node(
        "expand",
        custom_expand(tools, model=model, provider=provider),
    )
    builder.add_edge(START, "start")

    builder.add_conditional_edges(
        "start",
        # Either expand/rollout or finish
        should_loop,
        ["expand", END],
    )
    builder.add_conditional_edges(
        "expand",
        # Either continue to rollout or finish
        should_loop,
        ["expand", END],
    )

    graph = builder.compile(checkpointer=CHECKPOINTER)

    return graph
