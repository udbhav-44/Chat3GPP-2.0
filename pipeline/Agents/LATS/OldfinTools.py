"""
 This file contains list of all tools that can be used by the agents. 
"""
from typing import Type , Dict , List , Union, Tuple, Any, Optional
from pydantic import BaseModel, Field
import requests
import json
import re
import os 
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import logging
from langchain.tools import BaseTool, tool
import time
import urllib.parse 
import zipfile
import traceback
import shutil
import tempfile
import threading
import subprocess
from neo4j import GraphDatabase
import pandas as pd
import concurrent.futures


ERROR_LOG_FILE = "./error_logs.log"

# Step 1: Create a logger
logger = logging.getLogger('my_logger')
file_Handler = logging.FileHandler(ERROR_LOG_FILE)
logger.setLevel(logging.DEBUG)  # Set the base logging level
file_Handler.setLevel(logging.ERROR)  # Set the handler logging level
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger.addHandler(file_Handler)
def log_error(tool_name, error_message, additional_info=None):
    error_entry = {
        "tool" : tool_name,
        "error_message" : error_message,
        "timestamp" : datetime.now().isoformat(),
        "additional info" : additional_info or {}
    }
    logger.error(json.dumps(error_entry, indent=4))

def _detect_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pm2.config.js").exists() or (parent / "docker-compose.yml").exists():
            return parent
    return Path(__file__).resolve().parents[3]

REPO_ROOT = _detect_repo_root()
PIPELINE_ROOT = Path(
    os.getenv("PIPELINE_ROOT", Path(__file__).resolve().parents[2])
).resolve()
load_dotenv(PIPELINE_ROOT / ".env", override=False)
ARTIFACTS_DIR = Path(
    os.getenv("PIPELINE_ARTIFACTS_DIR", PIPELINE_ROOT / "output" / "artifacts")
).resolve()
RESULTS_CSV_PATH = os.path.join(ARTIFACTS_DIR, "global", "Results.csv")
RESULTS_COLUMNS = [
    "doc_id",
    "title",
    "source_path",
    "meeting_id",
    "release",
    "total_score",
    "boosted_score",
]
JINA_SCRAPE_URL = os.getenv("JINA_SCRAPE_URL", "http://localhost:3000")
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "login123")
RAG_GENERATE_URL = os.getenv("RAG_GENERATE_URL", "http://127.0.0.1:4005/generate")
RAG_STATS_URL = os.getenv("RAG_STATS_URL", "http://127.0.0.1:4004/v1/statistics")
RAG_RETRIEVE_URL = os.getenv("RAG_RETRIEVE_URL", "http://127.0.0.1:4004/v1/retrieve")
RAG_UPLOADS_DIR = os.getenv("RAG_UPLOADS_DIR", REPO_ROOT / "rag" / "uploads")
USER_UPLOADS_DIR = os.getenv(
    "RAG_USER_UPLOADS_DIR",
    Path(RAG_UPLOADS_DIR).parent / "user_uploads",
)
_results_lock = threading.Lock()
_results_lock_by_thread: Dict[str, threading.Lock] = {}

import contextvars
# Context variable to store the current model. 
# This must be set by the caller (Agent/SolveSubQuery) before invoking tools.
current_model_var = contextvars.ContextVar("current_model", default=os.getenv("LLM_MODEL", "deepseek-chat"))
current_user_var = contextvars.ContextVar("current_user_id", default=None)
current_thread_var = contextvars.ContextVar("current_thread_id", default=None)

def get_current_model():
    return current_model_var.get()

def set_current_model(model):
    return current_model_var.set(model)

def reset_current_model(token):
    current_model_var.reset(token)


def get_current_user_id():
    return current_user_var.get()


def set_current_user_id(user_id):
    return current_user_var.set(str(user_id) if user_id else None)


def reset_current_user_id(token):
    if token is not None:
        current_user_var.reset(token)

def _dir_has_files(path: Optional[str]) -> bool:
    if not path or not os.path.isdir(path):
        return False
    for _, _, files in os.walk(path):
        for name in files:
            if not name.startswith("."):
                return True
    return False


def has_user_uploads(user_id: Optional[str] = None) -> bool:
    if user_id is None:
        user_id = get_current_user_id()
    if user_id is None:
        return False
    user_id = str(user_id).strip()
    if not user_id:
        return False
    user_dir = os.path.join(USER_UPLOADS_DIR, user_id)
    return _dir_has_files(user_dir)


def _sanitize_thread_id(thread_id):
    if not thread_id:
        return None
    value = str(thread_id).strip()
    if not value:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)

def _get_artifact_dir(thread_id=None):
    safe_thread_id = _sanitize_thread_id(thread_id) or get_current_thread_id() or "global"
    path = os.path.join(ARTIFACTS_DIR, safe_thread_id)
    os.makedirs(path, exist_ok=True)
    return path


def get_current_thread_id():
    return current_thread_var.get()


def set_current_thread_id(thread_id):
    return current_thread_var.set(_sanitize_thread_id(thread_id))


def reset_current_thread_id(token):
    if token is not None:
        current_thread_var.reset(token)


def get_results_csv_path(thread_id=None):
    return os.path.join(_get_artifact_dir(thread_id), "Results.csv")


def get_process_log_path(thread_id=None):
    return os.path.join(_get_artifact_dir(thread_id), "ProcessLogs.md")


def _get_results_lock(thread_id=None) -> threading.Lock:
    key = _sanitize_thread_id(thread_id) or get_current_thread_id() or "global"
    with _results_lock:
        lock = _results_lock_by_thread.get(key)
        if lock is None:
            lock = threading.Lock()
            _results_lock_by_thread[key] = lock
    return lock


def _get_thread_uploads_key(thread_id=None, user_id=None) -> str:
    safe_thread_id = _sanitize_thread_id(thread_id) or get_current_thread_id() or "global"
    safe_user_id = _sanitize_thread_id(user_id) or "anonymous"
    return f"{safe_user_id}__{safe_thread_id}"


def _get_thread_uploads_dir(thread_id=None, user_id=None) -> str:
    uploads_key = _get_thread_uploads_key(thread_id=thread_id, user_id=user_id)
    path = os.path.join(RAG_UPLOADS_DIR, uploads_key)
    os.makedirs(path, exist_ok=True)
    return path


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "").strip())
    return cleaned.strip("_.") or "doc"


def _resolve_soffice_bin() -> str:
    for candidate in [os.getenv("SOFFICE_BIN"), "soffice", "libreoffice"]:
        if candidate and shutil.which(candidate):
            return candidate
    raise RuntimeError(
        "LibreOffice binary not found. Install libreoffice or set SOFFICE_BIN to the binary path."
    )


def _convert_legacy_word_to_docx(src_path: str, dst_docx_path: str) -> None:
    soffice_bin = _resolve_soffice_bin()
    with tempfile.TemporaryDirectory(prefix="word-convert-") as out_dir:
        # Saving as .docx strips VBA macros from .docm/.doc by design.
        cmd = [
            soffice_bin,
            "--headless",
            "--nologo",
            "--nolockcheck",
            "--nodefault",
            "--nofirststartwizard",
            "--convert-to",
            "docx:MS Word 2007 XML",
            "--outdir",
            out_dir,
            src_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(f"LibreOffice conversion failed (exit {proc.returncode}): {detail}")

        expected_name = os.path.splitext(os.path.basename(src_path))[0] + ".docx"
        converted_path = os.path.join(out_dir, expected_name)
        if not os.path.exists(converted_path):
            docx_candidates = [
                os.path.join(out_dir, name)
                for name in os.listdir(out_dir)
                if name.lower().endswith(".docx")
            ]
            if len(docx_candidates) == 1:
                converted_path = docx_candidates[0]
            else:
                raise RuntimeError(
                    f"Converted file not found in {out_dir}. stdout={proc.stdout.strip()!r}"
                )

        shutil.move(converted_path, dst_docx_path)


def _dedupe_results(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    def _resolve_column(base_name: str) -> Optional[str]:
        if base_name in df.columns:
            return base_name
        suffix = f".{base_name}".lower()
        for col in df.columns:
            if str(col).lower().endswith(suffix):
                return col
        return None

    score_col = _resolve_column("boosted_score") or _resolve_column("total_score")
    if score_col:
        df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
        df = df.sort_values(score_col, ascending=False, na_position="last")

    doc_id_col = _resolve_column("doc_id")
    source_col = _resolve_column("source_path")
    if doc_id_col:
        doc_series = df[doc_id_col].fillna("").astype(str).str.strip()
        has_doc_id = doc_series != ""
        with_doc_id = df[has_doc_id]
        without_doc_id = df[~has_doc_id]
        if not with_doc_id.empty:
            with_doc_id = with_doc_id.drop_duplicates(subset=[doc_id_col], keep="first")
        if source_col and not without_doc_id.empty:
            source_series = without_doc_id[source_col].fillna("").astype(str).str.strip()
            has_source = source_series != ""
            with_source = without_doc_id[has_source]
            without_source = without_doc_id[~has_source]
            if not with_source.empty:
                with_source = with_source.drop_duplicates(subset=[source_col], keep="first")
            return pd.concat([with_doc_id, with_source, without_source], ignore_index=True)
        return pd.concat([with_doc_id, without_doc_id], ignore_index=True)
    if source_col:
        return df.drop_duplicates(subset=[source_col], keep="first")
    return df.drop_duplicates()


def _align_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in RESULTS_COLUMNS if c in df.columns]
    extras = [c for c in df.columns if c not in cols]
    return df.reindex(columns=cols + extras)


def _normalize_results_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    result = df.copy()
    for base in RESULTS_COLUMNS:
        candidates = []
        for col in result.columns:
            if col == base:
                candidates.append(col)
                continue
            col_str = str(col)
            if col_str.lower().endswith(f".{base}".lower()):
                candidates.append(col)
        if not candidates:
            continue
        merged = None
        for col in candidates:
            series = result[col]
            if series.dtype == object:
                series = series.replace(r"^\s*$", pd.NA, regex=True)
            merged = series if merged is None else merged.combine_first(series)
        result[base] = merged
        to_drop = [col for col in candidates if col != base]
        if to_drop:
            result = result.drop(columns=to_drop)
    return result


def merge_results_csv(new_df: pd.DataFrame, thread_id=None) -> pd.DataFrame:
    csv_path = get_results_csv_path(thread_id)
    lock = _get_results_lock(thread_id)
    with lock:
        if os.path.exists(csv_path):
            try:
                existing_df = pd.read_csv(csv_path)
            except pd.errors.EmptyDataError:
                existing_df = pd.DataFrame(columns=RESULTS_COLUMNS)
        else:
            existing_df = pd.DataFrame(columns=RESULTS_COLUMNS)

        existing_df = _normalize_results_columns(existing_df)
        new_df = _normalize_results_columns(new_df) if new_df is not None else new_df

        if new_df is None or new_df.empty:
            if os.path.exists(csv_path):
                save_results_csv(existing_df, csv_path=csv_path)
            elif existing_df.empty:
                save_results_csv(existing_df, csv_path=csv_path)
            return existing_df

        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = _dedupe_results(combined)
        combined = _align_results_columns(combined)
        save_results_csv(combined, csv_path=csv_path)
        return combined



def save_results_csv(df, csv_path=None):
    csv_path = csv_path or get_results_csv_path()
    tmp_csv_path = f"{csv_path}.tmp"
    try:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(tmp_csv_path, index=False)
        os.replace(tmp_csv_path, csv_path)
        logging.info(f"Saved results atomically to {csv_path}")
        return True
    except Exception as e:
        log_error("search_and_generate:csv_save", str(e), {"csv_path": csv_path})
        try:
            if os.path.exists(tmp_csv_path):
                os.remove(tmp_csv_path)
        except OSError:
            pass
        return False


_g_api = os.getenv("GEMINI_API_KEY_30")
if _g_api:
    os.environ["GOOGLE_API_KEY"] = _g_api
_oai = os.getenv("OPENAI_API_KEY")
if _oai:
    os.environ["OPENAI_API_KEY"] = _oai
_tav = os.getenv("TAVILY_API_KEY")
if _tav:
    os.environ["TAVILY_API_KEY"] = _tav

TAVILY_API_URL = "https://api.tavily.com/search"

def clean_text(text):
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s.,!?\'"]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text



def _tavily_search_request(
    query: str,
    max_results: int,
    search_depth: str,
    include_answer: bool,
    include_raw_content: bool,
    include_images: bool,
) -> List[Dict[str, Any]]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        log_error(
            tool_name="tavily_web_search",
            error_message="Missing TAVILY_API_KEY",
            additional_info={"query": query},
        )
        return []

    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": search_depth,
        "include_answer": include_answer,
        "include_raw_content": include_raw_content,
        "include_images": include_images,
    }

    try:
        response = requests.post(TAVILY_API_URL, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data.get("results", [])
    except requests.RequestException as e:
        log_error(
            tool_name="tavily_web_search",
            error_message=f"{type(e).__name__}: {e}",
            additional_info={
                "query": query,
                # "status_code": getattr(getattr(e, "response", None), "status_code", None),
            },
        )
        return []
    except Exception as e:
        log_error(
            tool_name="tavily_web_search",
            error_message=f"{type(e).__name__}: {e}",
            additional_info={"query": query},
        )
        return []

class MyCustomToolInput(BaseModel):
    """Input schema for MyCustomTool."""
    argument: str = Field(..., description="Description of the argument.")

class MyCustomTool(BaseTool):
    name: str = "Name of my tool"
    description: str = (
        "Clear description for what this tool is useful for, you agent will need this information to use it."
    )
    args_schema: Type[BaseModel] = MyCustomToolInput

    def _run(self, argument: str) -> str:
        # Implementation goes here
        return "this is an example of a tool output, ignore it and move along."
    

@tool
def web_scrape(url, query) -> Union[Dict, str]:
    """
    Use this to scrape a web page using links found using web search to give detailed response. Input should be the URL of the page to scrape.
    Returns the scraped data as JSON if successful, else move on to the next best site in case of errors like required login, captcha etc.
    Args:
        url (str): The URL of the page to scrape.
        query (str): The query for which the page is being scraped.
    Returns:
        Union[Dict, str]: The scraped data as JSON if successful
    """
    print("web_scrape invoked")
    # JINA HOSTING - URL Change
    api_url = f"{JINA_SCRAPE_URL.rstrip('/')}/{url}"
    headers = {
        'Accept': 'application/json',
        'X-Respond-With':'markdown',
        
    }
    output_folder = 'temp_rag_space'
    try:
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate filename based on URL and timestamp
        filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        filepath = os.path.join(output_folder, filename)

        response = requests.get(api_url, headers=headers)
        response.raise_for_status()

        try:
            data = response.json()
            data_str = str(data)
        except ValueError:
            data_str = response.text
        finally:
            # Save the data to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(data_str)
            delay = 2
            time.sleep(delay)

            return query_documents.invoke({"prompt":query,"source":url})

    except requests.RequestException as e:
        log_error(
            tool_name="web_scrape",
            error_message=str(e),
            additional_info={"url": url}
        )
        return url
    
    

def list_directories(ftp_client, base_path):
    """
    Lists directories in the given base_path on the FTP server.
    """
    try:
        ftp_client.cwd(base_path)
        directories = ftp_client.nlst()
        logging.info("Available directories in %s: %s", base_path, directories)
        return directories
    except Exception as e:
        logging.exception("Error listing directories in %s", base_path)
        traceback.print_exc()
        return []
    
def clear_directory(path):

    """Clear all files and directories in the specified path"""
    logging.info(f"Clearing directory: {path}")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        return
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

def format_response(response_text):
    """Format JSON or text responses nicely"""
    logging.info(f"Formatting response: {response_text}")
    try:
        data = json.loads(response_text)
        formatted = "### Generated Response\n\n"
        if isinstance(data, dict):
            for k, v in data.items():
                formatted += f"**{k}**:\n{v}\n\n"
        elif isinstance(data, list):
            for i, item in enumerate(data, start=1):
                formatted += f"**Result {i}:**\n{json.dumps(item, indent=2)}\n\n"
        else:
            formatted = str(data)
        return formatted
    except json.JSONDecodeError:
        # Fallback: raw text as markdown
        return f"###  Generated Response\n\n{response_text}"


@tool
def search_and_generate(query_str: str, meeting_id: Optional[str] = "") -> Tuple[Optional[Any], str]:
    """
    Tool to get relevant documents from Graph3GPP based on a search query and generate a response using RAG service.
    Returns a tuple of (DataFrame of results or None, formatted response string).
    meeting_id is optional and can be used to filter documents by meeting.

    USE THIS TOOL INSTEAD OF query_documents FOR BEST RESULTS. IN CASE OF FAILURE, FALLBACK TO query_documents.

    Args:
        query_str (str): The search query string.
        meeting_id (Optional[str]): An optional meeting ID to filter documents.
    Returns:
        Tuple[Optional[Any], str]: A tuple containing a DataFrame of results (or None) and a formatted response string.

    CALL THIS TOOL FIRST BEFORE ANY OTHER TOOL.
    """
    output_dir = tempfile.mkdtemp(prefix="downloaded_docs_")
    uri = NEO4J_URI
    thread_id = get_current_thread_id()
    user_id = get_current_user_id()
    safe_thread_id = _sanitize_thread_id(thread_id) or "global"
    uploads_key = _get_thread_uploads_key(thread_id=safe_thread_id, user_id=user_id)
    uploads_dir = _get_thread_uploads_dir(thread_id=safe_thread_id, user_id=user_id)
    generate_uri = RAG_GENERATE_URL
    stats_uri = RAG_STATS_URL
    uname = NEO4J_USER
    pswd = NEO4J_PASSWORD

    logging.info(f"Received search request: {query_str}, {meeting_id}")

    try:

        os.makedirs(output_dir, exist_ok=True)

        query = """
        CALL () {
          CALL db.index.fulltext.queryNodes("docIndex", $query)
          YIELD node, score
          WHERE $meeting IS NULL OR node.meeting_id CONTAINS $meeting
          RETURN
            collect(node.doc_id) AS direct_doc_ids,
            collect({doc_id: node.doc_id, score: score}) AS direct_docs
        }
        WITH direct_doc_ids, direct_docs
        CALL (direct_doc_ids) {
          WITH direct_doc_ids
          CALL db.index.fulltext.queryNodes("agendaIndex", $query)
          YIELD node, score AS agenda_score
          MATCH (node)<-[:APPEARS_IN]-(d:Document)
          WITH d,
               CASE
                 WHEN d.doc_id IN direct_doc_ids THEN agenda_score * 2.3
                 ELSE agenda_score * 0.8
               END AS agenda_rel_score
          RETURN collect({doc_id: d.doc_id, score: agenda_rel_score}) AS agenda_docs
        }
        WITH direct_docs, agenda_docs
        CALL() {
          CALL db.index.fulltext.queryNodes("techEntityIndex", $query)
          YIELD node, score AS entity_score
          MATCH (d:Document)-[:MENTIONS]->(node)
          RETURN collect({doc_id: d.doc_id, score: entity_score * 0.7}) AS entity_docs
        }
        WITH direct_docs, agenda_docs, entity_docs
        WITH direct_docs + agenda_docs + entity_docs AS all_docs
        UNWIND all_docs AS doc_entry
        WITH doc_entry.doc_id AS doc_id, sum(doc_entry.score) AS total_score
        MATCH (d:Document {doc_id: doc_id})
        WITH d, total_score,
        CASE
          WHEN d.title CONTAINS 'Feature Lead Summary' THEN total_score * 2.0
          WHEN d.title CONTAINS 'Feature Lead' THEN total_score * 1.5
          ELSE total_score
        END AS boosted_score
        RETURN
          d.doc_id AS doc_id,
          d.title AS title,
          d.source_path AS source_path,
          d.meeting_id AS meeting_id,
          d.release AS release,
          total_score AS total_score,
          boosted_score AS boosted_score
        ORDER BY boosted_score DESC
        LIMIT 15;
        """

        driver = GraphDatabase.driver(uri, auth=(uname, pswd))
        logging.info(f"Connected to Neo4j: {uri}")

        meeting = meeting_id.strip() if meeting_id and meeting_id.strip() else None
        params = {"query": query_str, "meeting": meeting}

        with driver.session() as session:
            result = session.run(query, params)
            data = [record.data() for record in result]

        driver.close()
        logging.info(f"Found {len(data)} documents")

        if not data:
            empty_df = pd.DataFrame(columns=RESULTS_COLUMNS)
            merge_results_csv(empty_df, thread_id=thread_id)
            return empty_df, "⚠️ No matching documents found."

        df = pd.DataFrame(data)

        def download_and_extract(row):
            row_dict = row._asdict() if hasattr(row, "_asdict") else {}
            def pick_value(keys):
                for key in keys:
                    if key in row_dict:
                        value = row_dict.get(key)
                        if value not in (None, ""):
                            return value
                    if hasattr(row, key):
                        value = getattr(row, key)
                        if value not in (None, ""):
                            return value
                return ""

            doc_id = pick_value(["doc_id", "d.doc_id"])
            title = pick_value(["title", "d.title"])
            url = pick_value(["source_path", "d.source_path", "url"])
            if not doc_id or not url:
                logging.error("Missing document fields for download: %s", row_dict)
                return ("unknown", "missing doc_id/source_path", False)
            safe_title = str(title or "document")[:50].replace("/", "_")
            dest_path = os.path.join(output_dir, f"{doc_id} - {safe_title}.zip")
            safe_doc_id = _sanitize_filename(doc_id)
            doc_dir = os.path.join(uploads_dir, safe_doc_id)
            os.makedirs(doc_dir, exist_ok=True)
            if _dir_has_files(doc_dir):
                return (safe_title, None, False)
            temp_extract_dir = os.path.join(output_dir, safe_doc_id)
            os.makedirs(temp_extract_dir, exist_ok=True)

            try:
                encoded_url = urllib.parse.quote(url, safe=':/')
                r = requests.get(encoded_url, timeout=20)
                r.raise_for_status()
                with open(dest_path, "wb") as f:
                    f.write(r.content)

                with zipfile.ZipFile(dest_path, 'r') as zip_ref:
                    for member in zip_ref.namelist():
                        if not (member.startswith('__MACOSX/') or member.endswith('.DS_Store')):
                            zip_ref.extract(member, temp_extract_dir)
                os.remove(dest_path)

                for root, _, files in os.walk(temp_extract_dir):
                    for fname in files:
                        src_path = os.path.join(root, fname)
                        dst_path = os.path.join(doc_dir, fname)
                        if os.path.exists(dst_path):
                            continue

                        if fname.lower().endswith((".doc", ".docm")):
                            try:
                                clean_path = os.path.splitext(dst_path)[0] + ".docx"
                                if os.path.exists(clean_path):
                                    continue
                                _convert_legacy_word_to_docx(src_path, clean_path)
                                logging.info(f"Converted and moved safely: {clean_path}")
                            except Exception as e:
                                logging.error(f"LibreOffice conversion failed for {fname}: {e}")
                        else:
                            shutil.move(src_path, dst_path)

                shutil.rmtree(temp_extract_dir, ignore_errors=True)
                return (safe_title, None, True)
            except Exception as e:
                logging.error(f"Error downloading {safe_title}: {e}")
                return (safe_title, str(e), False)

        download_errors = []
        downloaded_any = False
        max_workers = min(20, len(df))

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_and_extract, row): row
                for row in df.itertuples(index=False, name="Row")
            }
            for future in concurrent.futures.as_completed(futures):
                title, err, did_add = future.result()
                if err:
                    download_errors.append(f"{title}: {err}")
                if did_add:
                    downloaded_any = True

        logging.info(f"Downloaded {len(df) - len(download_errors)}/{len(df)} documents successfully")

        if downloaded_any:
            # wait for AI service readiness only when new docs were added
            start_time = time.time()
            max_wait = int(os.getenv("RAG_STATS_MAX_WAIT", "300"))
            poll_interval = float(os.getenv("RAG_STATS_POLL_INTERVAL", "5"))
            ready = False
            while time.time() - start_time < max_wait:
                try:
                    resp = requests.get(stats_uri, timeout=5)
                    if resp.status_code == 200:
                        ready = True
                        logging.info("AI service ready.")
                        break
                    logging.info("AI not ready, status=%s", resp.status_code)
                except Exception as e:
                    logging.info("AI service check failed: %s", e)
                time.sleep(poll_interval)

            if not ready:
                msg = f"Timeout: AI service not ready after {max_wait/60:.1f} minutes."
                logging.error(msg)
                return df, msg

        # generate
        payload = {
            "query": query_str,
            "max_tokens": 5000,
            "num_docs": 10,
            "model": get_current_model(),
            "thread_id": uploads_key,
        }
        try:
            response = requests.post(generate_uri, json=payload, timeout=90)
            response.raise_for_status()
            formatted_response = format_response(json.dumps(response.json()))
        except Exception as e:
            formatted_response = f"Failed to generate response: {e}"
            log_error("search_and_generate", str(e), {"query": query_str, "meeting_id": meeting_id})

        merge_results_csv(df, thread_id=thread_id)


        return df, formatted_response

    except Exception as e:
        log_error("search_and_generate", str(e), {"query": query_str, "meeting_id": meeting_id})
        return None, f"Error: {e}"
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    



@tool
def web_search(query: str):
    """
    If you do not know about an entity, Perform web search using google search engine. 
    This should be followed by web scraping the most relevant page to get detailed response.
    Args:
        query (str): The query to search for.
    Returns:
        str: The URL of the most relevant page to scrape.
    """
    print("web_search invoked")
    try:
        res = []
        search_results = _tavily_search_request(
            query=query,
            max_results=2,
            search_depth="basic",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
        if not search_results:
            return ''
        try:
            for search_result in search_results:
                url = search_result.get("url")
                if not url:
                    continue
                res.append(web_scrape.invoke({"url": url, "query": query}))
            return res
        except Exception as e:
            # If both fail, return error message
            log_error(
                tool_name="tavily_web_search",
                error_message=str(e),
                additional_info={"query": query}
            )
            return search_results
    except Exception as e:
        # If both fail, return error message
        log_error(
            tool_name="tavily_web_search",
            error_message=str(e),
            additional_info={"query": query}
        )
        return ''

@tool
def web_search_simple(query: str):
    """
    If you do not know about an entity, Perform web search using google search engine.
    Args:
        query (str): The query to search for.
    Returns:
        str: The URL of the most relevant page to scrape.      
    """
    
    print("web_search_simple invoked")
    
    try:
        return _tavily_search_request(
            query=query,
            max_results=3,
            search_depth="advanced",
            include_answer=True,
            include_raw_content=False,
            include_images=False,
        )
    except Exception as e:
        log_error(
            tool_name="tavily_web_search_simple",
            error_message=str(e),
            additional_info={"query": query},
        )
        return ''


def _user_path_matches(metadata: Dict[str, Any], user_id: str) -> bool:
    path = str(metadata.get("path", "")).replace("\\", "/").lower()
    token = f"/user_uploads/{user_id}/".lower()
    return token in path or path.startswith(f"user_uploads/{user_id}/".lower())


def _filter_docs_for_user(docs: List[Dict[str, Any]], user_id: str) -> List[Dict[str, Any]]:
    if not user_id:
        return docs
    filtered = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
        if _user_path_matches(meta, user_id):
            filtered.append(doc)
    return filtered



@tool
def query_documents(prompt: str, source: str) -> Dict:
    """
    Query documents using a Retrieval-Augmented Generation (RAG) endpoint.
    This should be the first choice before doing web search,
    if this fails or returns unsatisfactory results, then use web search for the same query.
    Args:
        prompt (str): The prompt to send to the RAG endpoint.
        source (str): The source URL of the document.

    Returns:
        Dict: The JSON response from the RAG endpoint, containing the retrieved information and generated answer.
    """
    try:
        logging.info("query_documents started")
        start = time.time()
        
        user_id = get_current_user_id()
        payload = {
            "query": prompt,  # No need to quote the prompt
            "source": source,  # source should be a string, not a set
            "model": get_current_model()
        }
        if user_id:
            payload["user_id"] = user_id
        
        response = requests.post(
            RAG_GENERATE_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        logging.info("query_documents response status: %s", response.status_code)
        logging.info("query_documents posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        logging.info("query_documents response validated")
        
        end = time.time()
        logging.info("query_documents time taken: %.2fs", end - start)
        
        result = response.json()
        logging.debug("query_documents result: %s", result)
        return result
    
    except requests.RequestException as e:
        logging.exception("query_documents request failed")
        if hasattr(e, 'response'):
            logging.error("query_documents response status: %s", e.response.status_code)
            logging.error("query_documents response content: %s", e.response.text)
        log_error(
            tool_name="query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt, "source": source}
        )
        return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"



@tool
def simple_query_documents(prompt: str) -> Dict:
    """
    Query documents using a Retrieval-Augmented Generation (RAG) endpoint.
    This should be the first choice before doing web search,
    if this fails or returns unsatisfactory results, then use web search for the same query.

    Args:
        prompt (str): The prompt to send to the RAG endpoint.
        source (str): The source URL of the document.

    Returns:
        Dict: The JSON response from the RAG endpoint, containing the retrieved information and generated answer.
    """
    try:
        logging.info("simple_query_documents started")
        start = time.time()
        
        user_id = get_current_user_id()
        if not has_user_uploads(user_id):
            logging.info("simple_query_documents skipped: no user uploads for user_id=%s", user_id)
            return "No user uploads found. Ask the user to upload relevant documents."
        payload = {
            "query": prompt, 
            "destination": 'user',
            "model": get_current_model()
        }
        if user_id:
            payload["user_id"] = user_id
        logging.debug("simple_query_documents payload: %s", payload)
        response = requests.post(
            RAG_GENERATE_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        logging.info("simple_query_documents response status: %s", response.status_code)
        logging.info("simple_query_documents posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        logging.info("simple_query_documents response validated")
        
        end = time.time()
        logging.info("simple_query_documents time taken: %.2fs", end - start)
        
        result = response.json()
        logging.debug("simple_query_documents result: %s", result)
        return result
    
    except requests.RequestException as e:
        logging.exception("simple_query_documents request failed")
        if hasattr(e, 'response'):
            if e.response is not None:
                logging.error("simple_query_documents response status: %s", e.response.status_code)
                logging.error("simple_query_documents response content: %s", e.response.text)
        log_error(
            tool_name="simple_query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt}
        )
        
        return ''
        
@tool
def retrieve_documents(prompt: str) -> str:
    """
    Extract Information from the provided internal document
    Since this is the main source of information which is always 
    correct, this should be the first choice of tool for any agent.
    CALL THIS BEFORE CALLING ANY OTHER TOOL.
    Args:
        prompt (str): The prompt to send to the RAG endpoint.
        source (str): The source URL of the document.
    Returns:
        Dict: The JSON response from the RAG endpoint, containing the retrieved information and generated answer.
    """

    try:
        logging.info("retrieve_documents started")
        start = time.time()
        
        user_id = get_current_user_id()
        if not has_user_uploads(user_id):
            logging.info("retrieve_documents skipped: no user uploads for user_id=%s", user_id)
            return "No user uploads found. Ask the user to upload relevant documents."
        k = 2
        if user_id:
            k = 12
        payload = {
            "query": prompt,
            "k": k,
            "destination": 'user'
        }
        
        response = requests.post(
            RAG_RETRIEVE_URL,
            headers={"Content-Type": "application/json"},
            json=payload
        )
        
        logging.info("retrieve_documents response status: %s", response.status_code)
        logging.info("retrieve_documents posted")
        response.raise_for_status()  # Raise an error for HTTP issues
        logging.info("retrieve_documents response validated")
        
        end = time.time()
        logging.info("retrieve_documents time taken: %.2fs", end - start)
        
        result = response.json()
        if user_id:
            result = _filter_docs_for_user(result, user_id)
            if not result:
                return "No matching documents found for this user. Ask the user to upload relevant documents."
        out = ''
        for i in result:
            for j in i.values():
                if type(j) is str:
                    out += f"{j} "
                elif type(j) is dict:
                    for k in j.values():
                        out += f"{str(j)} "
                    out+= '\n'
            out+= '\n'
        return out
    
    except requests.RequestException as e:
        logging.exception("retrieve_documents request failed")
        if hasattr(e, 'response'):
            if e.response is not None:
                logging.error("retrieve_documents response status: %s", e.response.status_code)
                logging.error("retrieve_documents response content: %s", e.response.text)
        log_error(
            tool_name="query_documents",
            error_message=str(e),
            additional_info={"prompt": prompt}
        )
        return "This tool is not working right now. DO NOT CALL THIS TOOL AGAIN!"
        # return web_search_simple.invoke(prompt)
