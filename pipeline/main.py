"""
This script sets up a WebSocket server to handle various types of queries and tasks using different agents and APIs.
Functions:
    mainBackend(query, websocket, rag):
        Handles the main backend processing of queries, including classification, planning, and execution of tasks using various agents.
    handle_connection(websocket):
        Manages incoming WebSocket connections and routes messages to the appropriate handlers.
    main():
        Starts the WebSocket server and keeps it running indefinitely.
"""
import os
import shutil
import threading
from pathlib import Path
from dotenv import load_dotenv
import uuid
import logging
from logging_config import setup_logging

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
load_dotenv(ENV_PATH, override=False)
setup_logging()
logger = logging.getLogger("backend")

import time
import contextlib
import json
import re
import base64
import mimetypes
import hmac
import hashlib
from typing import Optional
# Import custom agents for different tasks
from Agents.Agents import Agent
from Agents.Smack import Smack
from Agents.ClassifierAgent import classifierAgent, classifierAgent_RAG
from Agents.PlannerAgent import plannerAgent, plannerAgent_rag
from Agents.DrafterAgent import drafterAgent_vanilla, drafterAgent_rag
from Agents.ConciseAnsAgent import conciseAns_vanilla
from Agents.RAG_Agent import ragAgent
from Agents.LATS.OldfinTools import *
from langchain_core.messages import HumanMessage, SystemMessage
from LLMs import get_llm
import asyncio
import websockets
from langchain.globals import set_verbose
from makeGraphJSON import makeGraphJSON

set_verbose(os.getenv("LANGCHAIN_VERBOSE", "false").lower() == "true")
now = time.time()
PROCESS_LOG_PATH = BASE_DIR / "ProcessLogs.md"
GRAPH_PATH = BASE_DIR / "Graph.json"
BAD_QUESTION_PATH = BASE_DIR / "Bad_Question.md"
OUTPUT_DIR = BASE_DIR / "output"
WRITE_ARTIFACTS = os.getenv("WRITE_ARTIFACTS", "false").lower() in {"1", "true", "yes"}
CHAT_IMAGE_UPLOADS_DIR = Path(
    os.getenv(
        "CHAT_IMAGE_UPLOADS_DIR",
        (BASE_DIR / ".." / "RAG" / "chat_uploads"),
    )
).resolve()
CHAT_IMAGE_MAX_BYTES = int(os.getenv("CHAT_IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))
CHAT_IMAGE_MAX_PER_MESSAGE = int(os.getenv("CHAT_IMAGE_MAX_PER_MESSAGE", "6"))
CHAT_IMAGE_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
VISION_MODEL = os.getenv("VISION_MODEL")
VISION_PROVIDER = os.getenv("VISION_PROVIDER")

JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
MAX_CONCURRENT_QUERIES = int(os.getenv("MAX_CONCURRENT_QUERIES", "4"))
MAX_ACTIVE_TASKS_PER_CONNECTION = int(os.getenv("MAX_ACTIVE_TASKS_PER_CONNECTION", "4"))
MAX_CONCURRENT_QUERIES_PER_USER = int(
    os.getenv("MAX_CONCURRENT_QUERIES_PER_USER", str(MAX_ACTIVE_TASKS_PER_CONNECTION))
)
WS_SEND_QUEUE_MAX = int(os.getenv("WS_SEND_QUEUE_MAX", "100"))
WS_MAX_SIZE = int(os.getenv("WS_MAX_SIZE", str(8 * 1024 * 1024)))
WS_MAX_QUEUE = int(os.getenv("WS_MAX_QUEUE", "32"))
WS_COMPRESSION = os.getenv("WS_COMPRESSION", "true").lower() in {"1", "true", "yes"}
_query_semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES)
_user_semaphores = {}
_user_semaphores_lock = threading.Lock()


def _get_user_semaphore(user_id: Optional[str]) -> asyncio.Semaphore:
    key = str(user_id or "anonymous")
    with _user_semaphores_lock:
        semaphore = _user_semaphores.get(key)
        if semaphore is None:
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_QUERIES_PER_USER)
            _user_semaphores[key] = semaphore
    return semaphore


class WebSocketOutbox:
    def __init__(self, websocket, maxsize=0):
        self.websocket = websocket
        self.queue = asyncio.Queue(maxsize=maxsize) if maxsize else asyncio.Queue()
        self._closed = asyncio.Event()
        self._task = asyncio.create_task(self._sender())

    async def _sender(self):
        try:
            while True:
                message = await self.queue.get()
                if message is None:
                    break
                await self.websocket.send(message)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._closed.set()

    async def send_json(self, payload: dict) -> bool:
        if self._closed.is_set():
            return False
        await self.queue.put(json.dumps(payload))
        return True

    async def close(self):
        if self._closed.is_set():
            return
        await self.queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._task

def should_abort(cancel_event):
    return cancel_event is not None and cancel_event.is_set()

async def run_blocking(func, *args, **kwargs):
    return await asyncio.to_thread(func, *args, **kwargs)

def _b64url_decode(segment: str) -> bytes:
    padding = "=" * (-len(segment) % 4)
    return base64.urlsafe_b64decode(segment + padding)

def _decode_jwt_payload(token: str) -> Optional[dict]:
    try:
        header_b64, payload_b64, signature_b64 = token.split(".")
        header = json.loads(_b64url_decode(header_b64))
        if header.get("alg") != "HS256":
            return None
        signing_input = f"{header_b64}.{payload_b64}".encode("utf-8")
        expected_sig = hmac.new(
            JWT_SECRET.encode("utf-8"),
            signing_input,
            hashlib.sha256,
        ).digest()
        actual_sig = _b64url_decode(signature_b64)
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64url_decode(payload_b64))
        exp = payload.get("exp")
        if exp is not None and time.time() > float(exp):
            return None
        return payload
    except Exception:
        return None

def _extract_user_from_token(token: Optional[str]) -> Optional[dict]:
    if not token or token.count(".") != 2:
        return None
    return _decode_jwt_payload(token)

def _get_checkpoint_conn_str():
    url = os.getenv("LANGGRAPH_CHECKPOINT_URL") or os.getenv("CHECKPOINT_DATABASE_URL")
    if url:
        return url
    user = os.getenv("POSTGRES_USER", "udbhav")
    password = os.getenv("POSTGRES_PASSWORD", "login123")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "postgres")
    return f"postgresql://{user}:{password}@{host}:{port}/{db}"

def ensure_checkpoint_tables() -> bool:
    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError:
        logger.warning(
            "langgraph-checkpoint-postgres not installed; Postgres checkpointing disabled."
        )
        return False

    conn_str = _get_checkpoint_conn_str()
    try:
        with PostgresSaver.from_conn_string(conn_str) as saver:
            if hasattr(saver, "setup"):
                saver.setup()
        logger.info("Postgres checkpoint tables are ready.")
        return True
    except Exception:
        logger.exception("Failed to initialize Postgres checkpointer.")
        return False

CHECKPOINT_READY = ensure_checkpoint_tables()

def _build_thread_filters(thread_id: str):
    base = str(thread_id)
    exact_ids = {base, f"concise:{base}"}
    prefixes = {f"{base}:%"}
    return exact_ids, prefixes

def delete_thread_state(thread_id: str) -> bool:
    try:
        import psycopg
    except ImportError:
        logger.warning("psycopg not installed; skipping thread delete for %s", thread_id)
        return False
    if not CHECKPOINT_READY:
        logger.info("Checkpoint tables not ready; skipping thread delete for %s", thread_id)
        return False

    conn_str = _get_checkpoint_conn_str()
    exact_ids, prefixes = _build_thread_filters(thread_id)
    tables = ("checkpoint_writes", "checkpoint_blobs", "checkpoints")
    deleted_rows = 0

    try:
        with psycopg.connect(conn_str) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT to_regclass(%s), to_regclass(%s), to_regclass(%s)",
                    ("checkpoint_writes", "checkpoint_blobs", "checkpoints"),
                )
                if not all(cur.fetchone() or []):
                    logger.info("Checkpoint tables missing; skipping delete for %s", thread_id)
                    return False
                for table in tables:
                    for tid in exact_ids:
                        cur.execute(
                            f"DELETE FROM {table} WHERE thread_id = %s",
                            (tid,),
                        )
                        deleted_rows += cur.rowcount
                    for prefix in prefixes:
                        cur.execute(
                            f"DELETE FROM {table} WHERE thread_id LIKE %s",
                            (prefix,),
                        )
                        deleted_rows += cur.rowcount
            conn.commit()
        logger.info("Deleted %s checkpoint rows for thread %s", deleted_rows, thread_id)
        return True
    except Exception:
        logger.exception("Failed to delete thread state for %s", thread_id)
        return False


def cleanup_thread_files(thread_id: str, user_id: Optional[str] = None) -> None:
    safe_thread_id = _sanitize_thread_id(thread_id) or str(thread_id)
    uploads_key = (
        _get_thread_uploads_key(thread_id=safe_thread_id, user_id=user_id)
        if user_id
        else safe_thread_id
    )
    if ARTIFACTS_DIR:
        artifact_path = os.path.join(ARTIFACTS_DIR, safe_thread_id)
        if os.path.isdir(artifact_path):
            shutil.rmtree(artifact_path, ignore_errors=True)
    if RAG_UPLOADS_DIR:
        upload_path = os.path.join(RAG_UPLOADS_DIR, uploads_key)
        if os.path.isdir(upload_path):
            shutil.rmtree(upload_path, ignore_errors=True)
    if CHAT_IMAGE_UPLOADS_DIR and user_id:
        safe_user = _sanitize_user_id(user_id)
        if safe_user:
            image_dir = (CHAT_IMAGE_UPLOADS_DIR / safe_user / safe_thread_id).resolve()
            if image_dir.is_dir():
                shutil.rmtree(image_dir, ignore_errors=True)

async def mainBackend(
    query,
    send_json,
    rag,
    model=None,
    provider=None,
    allow_web_tools=False,
    cancel_event=None,
    thread_id=None,
    user_id=None,
    response_id=None,
    images=None,
):
    """
    Main backend function to process queries and interact with a websocket.
    This function handles different types of queries (simple or complex) and 
    processes them using various agents and pipelines. The function 
    also generates and sends responses back through the websocket.
    Args:
        query (str): The input query to be processed.
        websocket (WebSocket): The websocket connection to send responses.
        rag (bool): Flag to indicate if RAG mode is enabled.
    Returns:
        None
    """

    if should_abort(cancel_event):
        return
    user_token = None
    thread_token = None
    try:
        user_token = set_current_user_id(user_id)
    except Exception:
        logger.exception("Failed to set user context")
    try:
        thread_token = set_current_thread_id(thread_id)
    except Exception:
        logger.exception("Failed to set thread context")
    logger.info("Running mainBackend: %s", query)
    if model and "gpt-5" in str(model).lower():
        logger.warning("GPT-5 is disabled; falling back to default model.")
        model = None
    GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY_30')
    OPENAI_API_KEY = os.getenv('OPEN_AI_API_KEY_30')
    if WRITE_ARTIFACTS:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    LLM = 'OPENAI'
    key_dict = {
        'OPENAI': OPENAI_API_KEY,
        'GEMINI': GOOGLE_API_KEY
    }
    api_key = key_dict[LLM]

    IS_RAG = rag

    if IS_RAG == True:
        logger.info("RAG is ON")
    else:
        logger.info("RAG is OFF")

    if WRITE_ARTIFACTS:
        process_log_path = get_process_log_path(thread_id)
        with open(process_log_path, "w") as f:
            f.write("")

    resp = ''

    if should_abort(cancel_event):
        return

    image_paths = _resolve_image_paths(images, user_id, thread_id)
    if image_paths:
        logger.info("Processing %s image(s) for thread %s", len(image_paths), thread_id)
        if should_abort(cancel_event):
            return
        image_summary = await _summarize_images(query, image_paths, model=model, provider=provider)
        if should_abort(cancel_event):
            return
        if image_summary:
            query = f"{query}\n\nImage context:\n{image_summary}"
    web_tool_names = {"web_search", "web_scrape", "web_search_simple"}

    def filter_web_tools(tool_names):
        if allow_web_tools:
            return tool_names
        return [tool for tool in tool_names if tool not in web_tool_names]

    if not IS_RAG:
        logger.info("Running without internal docs context")
        has_uploads = has_user_uploads(get_current_user_id())
        if should_abort(cancel_event):
            return
        raw_query_type = await run_blocking(classifierAgent, query, model=model, provider=provider)
        query_type = raw_query_type.lower().strip()
        if "simple" in query_type:
            query_type = "simple"
        elif "complex" in query_type:
            query_type = "complex"
        else:
            logger.warning("Unexpected classifier output '%s'; defaulting to simple.", raw_query_type)
            query_type = "simple"
        if query_type == "complex":
            logger.info("Running complex task pipeline")
            if should_abort(cancel_event):
                return
            
            # plan -> dict
            plan = await run_blocking(
                plannerAgent,
                query,
                model=model,
                provider=provider,
                allow_web_tools=allow_web_tools,
            )
            if should_abort(cancel_event):
                return
            #This is the dictionary for UI Graph Construction
            dic_for_UI_graph = makeGraphJSON(plan['sub_tasks'])
            for node in dic_for_UI_graph['nodes']:
                tools = node['metadata'].get('tools', [])
                if not has_uploads:
                    tools = [
                        tool for tool in tools
                        if tool not in {"simple_query_documents", "retrieve_documents", "query_documents"}
                    ]
                if "search_and_generate" not in tools:
                    tools.insert(0, "search_and_generate")
                node['metadata']['tools'] = tools
            logger.debug("Graph payload: %s", dic_for_UI_graph)
            if should_abort(cancel_event):
                return
            await send_json({
                "type": "graph",
                "response": json.dumps(dic_for_UI_graph),
                "thread_id": thread_id,
                "response_id": response_id,
            })
            if WRITE_ARTIFACTS:
                with open(GRAPH_PATH, 'w') as fp:
                    json.dump(dic_for_UI_graph, fp)
            
            out_str = ''''''
            agentsList = []
            
            for sub_task in plan['sub_tasks']:
                if should_abort(cancel_event):
                    return
                agent_name = plan['sub_tasks'][sub_task]['agent']
                agent_role = plan['sub_tasks'][sub_task]['agent_role_description']
                local_constraints = plan['sub_tasks'][sub_task]['local_constraints']
                task = plan['sub_tasks'][sub_task]['content']
                dependencies = plan['sub_tasks'][sub_task]['require_data']
                tools_list = filter_web_tools(plan['sub_tasks'][sub_task]['tools'])
                if not has_uploads:
                    tools_list = [
                        tool for tool in tools_list
                        if tool not in {"simple_query_documents", "retrieve_documents", "query_documents"}
                    ]
                if "search_and_generate" not in tools_list:
                    tools_list.insert(0, "search_and_generate")
                agent_state = 'vanilla'
                logger.info("Processing agent: %s", agent_name)
                agent = Agent(
                    sub_task,
                    agent_name,
                    agent_role,
                    local_constraints,
                    task,
                    dependencies,
                    tools_list,
                    agent_state,
                    thread_id=thread_id,
                    model=model,
                    provider=provider,
                    allow_web_tools=allow_web_tools,
                )
                agentsList.append(agent)
            
            # Execute the task results using the Smack agent
            smack = Smack(agentsList)
            taskResultsDict = await run_blocking(smack.executeSmack)
            if should_abort(cancel_event):
                return
            for task in taskResultsDict:
                out_str += f'{taskResultsDict[task]} \n'
            resp = await run_blocking(drafterAgent_vanilla, query, out_str, model=model, provider=provider)
            if should_abort(cancel_event):
                return
            resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
            # resp = generate_chart(resp)
        elif query_type == "simple":
            logger.info("Running simple task pipeline")
            async def executeSimplePipeline(query):
                tools_list = [web_search_simple] if allow_web_tools else []
                resp = await run_blocking(
                    conciseAns_vanilla,
                    query,
                    tools_list,
                    thread_id=thread_id,
                    model=model,
                    provider=provider,
                )
                resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
                return str(resp)

            resp = await executeSimplePipeline(query)
            if should_abort(cancel_event):
                return

    elif IS_RAG:
        logger.info("Running internal docs RAG")
        has_uploads = has_user_uploads(get_current_user_id())
        rag_context = await run_blocking(ragAgent, query, state="concise", model=model, provider=provider)
        if should_abort(cancel_event):
            return
        raw_query_type = await run_blocking(
            classifierAgent_RAG,
            query,
            rag_context,
            model=model,
            provider=provider,
        )
        query_type = raw_query_type.lower().strip()
        if "simple" in query_type:
            query_type = "simple"
        elif "complex" in query_type:
            query_type = "complex"
        else:
            logger.warning("Unexpected RAG classifier output '%s'; defaulting to simple.", raw_query_type)
            query_type = "simple"
        logger.info("RAG query type: %s", query_type)
        
        if query_type == "complex":
            agent_state = 'RAG'
            logger.info("Running complex task pipeline")

            rag_context = await run_blocking(ragAgent, query, state="report", model=model, provider=provider)
            if should_abort(cancel_event):
                return
            plan = await run_blocking(
                plannerAgent_rag,
                query,
                rag_context,
                model=model,
                provider=provider,
                allow_web_tools=allow_web_tools,
            )
            if should_abort(cancel_event):
                return
            
            dic_for_UI_graph = makeGraphJSON(plan['sub_tasks'])
            for node in dic_for_UI_graph['nodes']:
                if has_uploads:
                    node['metadata']['tools'].append('retrieve_documents')
                else:
                    node['metadata']['tools'].append('search_and_generate')

            logger.debug("Graph payload: %s", dic_for_UI_graph)
            if should_abort(cancel_event):
                return
            await send_json({
                "type": "graph",
                "response": json.dumps(dic_for_UI_graph),
                "thread_id": thread_id,
                "response_id": response_id,
            })
            if WRITE_ARTIFACTS:
                with open(GRAPH_PATH, 'w') as fp:
                    json.dump(dic_for_UI_graph, fp)
            
            out_str = ''''''
            agentsList = []
            
            for sub_task in plan['sub_tasks']:
                if should_abort(cancel_event):
                    return
                agent_name = plan['sub_tasks'][sub_task]['agent']
                agent_role = plan['sub_tasks'][sub_task]['agent_role_description']
                local_constraints = plan['sub_tasks'][sub_task]['local_constraints']
                task = plan['sub_tasks'][sub_task]['content']
                dependencies = plan['sub_tasks'][sub_task]['require_data']
                tools_list = filter_web_tools(plan['sub_tasks'][sub_task]['tools'])
                if not has_uploads:
                    tools_list = [
                        tool for tool in tools_list
                        if tool not in {"simple_query_documents", "retrieve_documents"}
                    ]
                if "search_and_generate" not in tools_list:
                    tools_list.insert(0, "search_and_generate")
                logger.info("Processing agent: %s", agent_name)
                agent = Agent(
                    sub_task,
                    agent_name,
                    agent_role,
                    local_constraints,
                    task,
                    dependencies,
                    tools_list,
                    agent_state,
                    thread_id=thread_id,
                    model=model,
                    provider=provider,
                    allow_web_tools=allow_web_tools,
                )
                agentsList.append(agent)
            
            # Execute the task results using the Smack agent
            smack = Smack(agentsList)
            taskResultsDict = await run_blocking(smack.executeSmack)
            if should_abort(cancel_event):
                return
            for task in taskResultsDict:
                out_str += f'{taskResultsDict[task]} \n'
            resp = await run_blocking(
                drafterAgent_rag,
                query,
                rag_context,
                out_str,
                model=model,
                provider=provider,
            )
            if should_abort(cancel_event):
                return
            resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
            # resp = generate_chart(resp)
            if WRITE_ARTIFACTS:
                with open(OUTPUT_DIR / 'drafted_response.md', 'w') as f:
                    f.write(str(resp))

        elif query_type == 'simple':
            logger.info("Running simple task pipeline")
            resp = rag_context
            resp = re.sub(r'\\\[(.*?)\\\]', lambda m: f'$${m.group(1)}$$', resp, flags=re.DOTALL)
            if should_abort(cancel_event):
                return
                
    if should_abort(cancel_event):
        return
    await send_json({
        "type": "response",
        "response": resp,
        "thread_id": thread_id,
        "response_id": response_id,
    })
    if user_token is not None:
        try:
            reset_current_user_id(user_token)
        except Exception:
            logger.exception("Failed to reset user context")
    if thread_token is not None:
        try:
            reset_current_thread_id(thread_token)
        except Exception:
            logger.exception("Failed to reset thread context")


async def handle_connection(websocket):
    """
    Manages incoming WebSocket connections and routes messages to the appropriate handlers.
    
    Args:
        websocket (WebSocket): The WebSocket connection to manage.
    
    Returns:
        None: The function processes messages asynchronously and sends back responses to the client.
    """
    rag = False
    allow_web_tools = False
    active_tasks = {}
    active_cancel_events = {}
    connection_thread_id = f"conn-{uuid.uuid4().hex}"
    connection_user_id = None
    outbox = WebSocketOutbox(websocket, maxsize=WS_SEND_QUEUE_MAX)

    async def start_query(data):
        nonlocal active_tasks, active_cancel_events
        nonlocal connection_thread_id, connection_user_id
        if len(active_tasks) >= MAX_ACTIVE_TASKS_PER_CONNECTION:
            await outbox.send_json(
                {
                    "type": "error",
                    "error": "Too many active queries. Please wait.",
                }
            )
            return

        thread_id = data.get("thread_id")
        if thread_id:
            connection_thread_id = str(thread_id)
        else:
            thread_id = connection_thread_id

        user_payload = data.get("user")
        user_id = data.get("user_id")
        token_payload = _extract_user_from_token(data.get("auth_token"))
        if token_payload:
            user_id = token_payload.get("sub") or token_payload.get("email")
        if not user_id and isinstance(user_payload, dict):
            user_id = user_payload.get("id") or user_payload.get("email")
        if user_id:
            connection_user_id = str(user_id)
        else:
            user_id = connection_user_id

        active_cancel_event = asyncio.Event()
        response_id = data.get("response_id") or f"resp-{uuid.uuid4().hex}"

        async def runner():
            try:
                web_tools_flag = allow_web_tools
                if "web_tools" in data:
                    web_tools_flag = bool(data.get("web_tools"))
                user_semaphore = _get_user_semaphore(user_id)
                async with _query_semaphore, user_semaphore:
                    await mainBackend(
                        data['query'],
                        outbox.send_json,
                        rag,
                        model=data.get("model"),
                        provider=data.get("provider"),
                        allow_web_tools=web_tools_flag,
                        cancel_event=active_cancel_event,
                        thread_id=thread_id,
                        user_id=user_id,
                        response_id=response_id,
                        images=data.get("images"),
                    )
            except asyncio.CancelledError:
                return
            finally:
                active_cancel_events.pop(response_id, None)
                active_tasks.pop(response_id, None)

        active_cancel_events[response_id] = active_cancel_event
        active_tasks[response_id] = asyncio.create_task(runner())

    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Invalid websocket payload: %s", message)
                continue
            message_type = data.get("type")
            if message_type == 'query':
                logger.info("Received query: %s", data.get("query"))
                await start_query(data)

            if message_type == 'abort':
                target_id = data.get("response_id")
                if target_id:
                    cancel_event = active_cancel_events.get(target_id)
                    task = active_tasks.get(target_id)
                    if cancel_event:
                        cancel_event.set()
                    if task and not task.done():
                        task.cancel()
                else:
                    for cancel_event in list(active_cancel_events.values()):
                        cancel_event.set()
                    for task in list(active_tasks.values()):
                        if not task.done():
                            task.cancel()

            if message_type == 'delete_thread':
                target_thread_id = data.get("thread_id") or connection_thread_id
                logger.info("Deleting thread: %s", target_thread_id)
                for cancel_event in list(active_cancel_events.values()):
                    cancel_event.set()
                for task in list(active_tasks.values()):
                    if not task.done():
                        task.cancel()
                success = delete_thread_state(str(target_thread_id))
                cleanup_thread_files(str(target_thread_id), user_id=connection_user_id)
                if target_thread_id == connection_thread_id:
                    connection_thread_id = f"conn-{uuid.uuid4().hex}"
                await outbox.send_json(
                    {
                        "type": "thread_deleted",
                        "thread_id": str(target_thread_id),
                        "success": success,
                    }
                )

            if message_type == 'toggleRag':
                logger.info("Received toggleRag")
                if "query" in data:
                    rag = bool(data["query"])

            if message_type == 'toggleWebTools':
                logger.info("Received toggleWebTools")
                if "query" in data:
                    allow_web_tools = bool(data["query"])
                else:
                    rag = not rag
    except websockets.exceptions.ConnectionClosed as exc:
        logger.info("WebSocket closed: %s", exc)
    except Exception:
        logger.exception("Unexpected error in websocket handler")
    finally:
        for cancel_event in list(active_cancel_events.values()):
            cancel_event.set()
        for task in list(active_tasks.values()):
            if not task.done():
                task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await outbox.close()


def _get_env_seconds(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip().lower()
    if value in {"none", "null", "disabled", "false", "0"}:
        return None
    try:
        return float(value)
    except ValueError:
        logger.warning("Invalid %s=%r; using default %s", name, value, default)
        return default


def _sanitize_user_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", str(value).strip())
    return cleaned or None


def _sanitize_thread_id(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", str(value).strip())
    return cleaned or None


def _resolve_image_paths(images, user_id, thread_id):
    if not images or not user_id or not thread_id:
        return []
    safe_user = _sanitize_user_id(user_id)
    safe_thread = _sanitize_thread_id(thread_id) or str(thread_id)
    if not safe_user or not safe_thread:
        return []
    base_dir = (CHAT_IMAGE_UPLOADS_DIR / safe_user / safe_thread).resolve()
    if not base_dir.exists():
        return []
    resolved = []
    for item in list(images)[:CHAT_IMAGE_MAX_PER_MESSAGE]:
        filename = None
        if isinstance(item, dict):
            filename = item.get("filename")
        elif isinstance(item, str):
            filename = item
        if not filename:
            continue
        safe_name = Path(filename).name
        path = (base_dir / safe_name).resolve()
        if base_dir not in path.parents and path != base_dir:
            continue
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in CHAT_IMAGE_ALLOWED_EXT:
            continue
        try:
            if path.stat().st_size > CHAT_IMAGE_MAX_BYTES:
                continue
        except OSError:
            continue
        resolved.append(path)
    return resolved


def _guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _build_image_prompt(query: str, image_paths: list[Path]) -> list:
    parts = [
        {
            "type": "text",
            "text": (
                "You are a precise visual analyst. Describe the images in detail, "
                "focusing on information that helps answer the user question.\n"
                f"User question: {query}"
            ),
        }
    ]
    for path in image_paths:
        try:
            with open(path, "rb") as handle:
                data = handle.read()
        except OSError:
            continue
        mime_type = _guess_mime_type(path)
        b64 = base64.b64encode(data).decode("ascii")
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{b64}"},
            }
        )
    return parts


async def _summarize_images(query: str, image_paths: list[Path], model=None, provider=None) -> str:
    if not image_paths:
        return ""
    vision_model = VISION_MODEL or model
    vision_provider = VISION_PROVIDER or provider
    llm = get_llm(model=vision_model, provider=vision_provider, temperature=0.2, top_p=0.2)
    messages = [
        SystemMessage(
            content=(
                "Summarize the visual content clearly. "
                "If there is text in the image, transcribe it. "
                "If charts/tables exist, describe key values and trends."
            )
        ),
        HumanMessage(content=_build_image_prompt(query, image_paths)),
    ]
    try:
        response = await run_blocking(llm.invoke, messages)
    except Exception:
        logger.exception("Failed to summarize images")
        return ""
    return response.content if hasattr(response, "content") else str(response)


def maybe_enable_uvloop():
    if os.getenv("USE_UVLOOP", "true").lower() not in {"1", "true", "yes"}:
        return
    try:
        import uvloop  # type: ignore
    except Exception:
        logger.warning("uvloop not available; using default asyncio loop")
        return
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    logger.info("uvloop enabled")

async def main():
    """       
     Starts the WebSocket server and keeps it running indefinitely.
    """
    logger.info("WebSocket server starting on ws://0.0.0.0:8080")
    ping_interval = _get_env_seconds("WS_PING_INTERVAL", 20)
    ping_timeout = _get_env_seconds("WS_PING_TIMEOUT", 20)
    close_timeout = _get_env_seconds("WS_CLOSE_TIMEOUT", 10)
    compression = "deflate" if WS_COMPRESSION else None
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8080,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        close_timeout=close_timeout,
        max_size=WS_MAX_SIZE,
        max_queue=WS_MAX_QUEUE,
        compression=compression,
    ):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        maybe_enable_uvloop()
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown by user")
