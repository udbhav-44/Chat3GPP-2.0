from Agents.LATS.Create_graph import generateGraph_forLATS
from Agents.LATS.OldfinTools import *
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
import uuid
import logging
from LLMs import resolve_llm_settings
load_dotenv('.env')
logger = logging.getLogger(__name__)

def SolveSubQuery(query: str, tools, thread_id=None, checkpoint_ns=None, model=None, provider=None):
    question = query
    last_step = None
    _, resolved_model = resolve_llm_settings(
        model=model,
        provider=provider,
        role="lats",
    )
    token = set_current_model(resolved_model)
    graph = generateGraph_forLATS(
        tools,
        model=resolved_model,
        provider=provider,
    )
    checkpointer = getattr(graph, "checkpointer", None)
    if checkpointer is None:
        checkpointer = getattr(graph, "_checkpointer", None)
    has_checkpointer = checkpointer is not None
    config = None
    if has_checkpointer:
        thread_key = str(thread_id) if thread_id else f"lats-{uuid.uuid4().hex}"
        if checkpoint_ns:
            thread_key = f"{thread_key}:{checkpoint_ns}"
        config = {"configurable": {"thread_id": thread_key}}

    messages = None
    if config:
        try:
            snapshot = graph.get_state(config)
            if hasattr(snapshot, "values") and isinstance(snapshot.values, dict):
                messages = snapshot.values.get("messages")
            elif isinstance(snapshot, dict):
                messages = snapshot.get("messages")
        except Exception:
            messages = None

    input_state = {"input": question}
    if messages:
        input_state["messages"] = messages

    if config:
        iterator = graph.stream(input_state, config=config)
    else:
        iterator = graph.stream(input_state)

    for step in iterator:
        last_step = step
        step_name, step_state = next(iter(step.items()))
        logger.debug("Step: %s, height=%s", step_name, step_state["root"].height)
    try:
        solution_node = last_step["expand"]["root"].get_best_solution()
    except Exception as e:
        solution_node = last_step["start"]["root"].get_best_solution()
    best_trajectory = solution_node.get_trajectory(include_reflections=False)
    answer = best_trajectory[-1].content
    if config:
        try:
            graph.update_state(
                config,
                {"messages": [HumanMessage(content=question), AIMessage(content=answer)]},
                as_node="start",
            )
        except Exception:
            logger.exception("LATS memory update failed")

    reset_current_model(token)
    return answer
   
