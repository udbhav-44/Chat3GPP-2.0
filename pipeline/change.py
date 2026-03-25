"""
WebSocket server that watches files and streams updates to clients.

- output/artifacts/<thread>/ProcessLogs.md → plain appended logs (NO diff format)
- output/artifacts/<thread>/Results.csv     → structured table for sidebar
"""

import time
import os
import json
import threading
import asyncio
import websockets
import logging
import contextlib
from logging_config import setup_logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
from pathlib import Path
import csv
from websockets.connection import State

BASE_DIR = Path(__file__).resolve().parent
ARTIFACTS_DIR = Path(
    os.getenv("PIPELINE_ARTIFACTS_DIR") or (BASE_DIR / "output" / "artifacts")
).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
setup_logging()
logger = logging.getLogger("watcher")
AGENT_WS_EVENT_QUEUE_MAX = int(os.getenv("AGENT_WS_EVENT_QUEUE_MAX", "1000"))
AGENT_WS_CLIENT_QUEUE_MAX = int(os.getenv("AGENT_WS_CLIENT_QUEUE_MAX", "200"))
AGENT_WS_CLIENT_HIGH_QUEUE_MAX = int(
    os.getenv("AGENT_WS_CLIENT_HIGH_QUEUE_MAX", str(max(20, AGENT_WS_CLIENT_QUEUE_MAX // 4)))
)
AGENT_WS_COALESCE_MS = int(os.getenv("AGENT_WS_COALESCE_MS", "50"))
AGENT_WS_MAX_SIZE = int(os.getenv("AGENT_WS_MAX_SIZE", str(8 * 1024 * 1024)))
AGENT_WS_COMPRESSION = os.getenv("AGENT_WS_COMPRESSION", "true").lower() in {
    "1",
    "true",
    "yes",
}
clients = set()
clients_lock = asyncio.Lock()
message_queue = queue.Queue(maxsize=AGENT_WS_EVENT_QUEUE_MAX)
observer_started = threading.Event()
observer_thread = None


def _normalize_thread_ids(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v not in (None, "")]
    return [str(value)]


class ClientState:
    def __init__(self, websocket):
        self.websocket = websocket
        self.high_queue = asyncio.Queue(maxsize=AGENT_WS_CLIENT_HIGH_QUEUE_MAX)
        self.low_queue = asyncio.Queue(maxsize=AGENT_WS_CLIENT_QUEUE_MAX)
        self.subscriptions = set()
        self.subscribed = False
        self._closed = asyncio.Event()
        self._sender_task = asyncio.create_task(self._send_loop())

    async def _send_loop(self):
        try:
            while True:
                payload = None
                if not self.high_queue.empty():
                    payload = await self.high_queue.get()
                else:
                    payload = await self.low_queue.get()
                if payload is None:
                    break
                await self.websocket.send(payload)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self._closed.set()

    def matches(self, event):
        if not self.subscribed:
            return True
        if "*" in self.subscriptions:
            return True
        if not self.subscriptions:
            return False
        return str(event.get("thread_id")) in self.subscriptions

    def is_closed(self):
        return self.websocket.state is State.CLOSED or self._closed.is_set()

    def try_enqueue(self, payload, event_type):
        if self._closed.is_set():
            return False
        is_high = event_type in {"results"}
        target_queue = self.high_queue if is_high else self.low_queue
        try:
            target_queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            if is_high:
                try:
                    _ = target_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                try:
                    target_queue.put_nowait(payload)
                    return True
                except asyncio.QueueFull:
                    logger.warning("Dropping %s event for slow client", event_type)
                    return False
            logger.warning("Dropping %s event for slow client", event_type)
            return False

    async def close(self):
        if self._closed.is_set():
            return
        await self.high_queue.put(None)
        await self.low_queue.put(None)
        with contextlib.suppress(asyncio.CancelledError):
            await self._sender_task


def _enqueue_event(payload):
    try:
        message_queue.put_nowait(payload)
    except queue.Full:
        logger.warning("Dropping %s event: event queue full", payload.get("type"))


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

# --------------------------------------------------
# File System Event Handler
# --------------------------------------------------

class MyHandler(FileSystemEventHandler):
    def __init__(self):
        super().__init__()
        # Track how much of each log file we've already read
        self.log_offsets = {}

    def on_modified(self, event):
        self._handle_event(event.src_path)

    def on_created(self, event):
        self._handle_event(event.src_path)

    def on_moved(self, event):
        self._handle_event(event.dest_path)

    def _handle_event(self, src_path):
        resolved_path = Path(src_path).resolve()
        file_info = self._classify_path(resolved_path)
        if not file_info:
            return
        file_type, thread_id = file_info

        if file_type == "logs":
            self.handle_logs(resolved_path, thread_id)

        elif file_type == "results":
            self.handle_results(resolved_path, thread_id)

    def _classify_path(self, path: Path):
        name = path.name
        if name in {"ProcessLogs.md", "Results.csv"} and ARTIFACTS_DIR in path.parents:
            thread_id = None
            parent = path.parent
            if parent != ARTIFACTS_DIR:
                thread_id = parent.name or None
            if thread_id == "global":
                thread_id = None
            return ("logs" if name == "ProcessLogs.md" else "results", thread_id)
        if name == "ProcessLogs.md":
            return ("logs", None)
        if name.startswith("ProcessLogs-") and name.endswith(".md"):
            thread_id = name[len("ProcessLogs-"):-3] or None
            return ("logs", None if thread_id == "global" else thread_id)
        if name == "Results.csv":
            return ("results", None)
        if name.startswith("Results-") and name.endswith(".csv"):
            thread_id = name[len("Results-"):-4] or None
            return ("results", None if thread_id == "global" else thread_id)
        return None

    # --------------------------------------------------
    # ProcessLogs.md → plain tail-style logs
    # --------------------------------------------------
    def handle_logs(self, path: Path, thread_id):
        try:
            # Handle file truncation / rewrite
            offset = self.log_offsets.get(path, 0)
            if path.stat().st_size < offset:
                offset = 0

            with open(path, "r") as f:
                f.seek(offset)
                new_content = f.read()
                self.log_offsets[path] = f.tell()

            if new_content.strip():
                _enqueue_event({
                    "type": "logs",
                    "response": new_content,
                    "thread_id": thread_id
                })

        except FileNotFoundError:
            pass

    # --------------------------------------------------
    # Results.csv → sidebar table
    # --------------------------------------------------
    def handle_results(self, path: Path, thread_id):
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            _enqueue_event({
                "type": "results",
                "format": "table",
                "columns": reader.fieldnames,
                "rows": rows,
                "thread_id": thread_id
            })

        except FileNotFoundError:
            pass


# --------------------------------------------------
# Observer Thread
# --------------------------------------------------

def start_observer():
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path=str(ARTIFACTS_DIR), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


# --------------------------------------------------
# WebSocket Handler
# --------------------------------------------------

def ensure_observer_started():
    global observer_thread
    if observer_started.is_set():
        return
    observer_thread = threading.Thread(
        target=start_observer,
        daemon=True,
    )
    observer_thread.start()
    observer_started.set()

async def broadcast_loop():
    while True:
        event_data = await asyncio.to_thread(message_queue.get)
        if not event_data:
            continue
        if AGENT_WS_COALESCE_MS > 0:
            await asyncio.sleep(AGENT_WS_COALESCE_MS / 1000)
        events = [event_data]
        while True:
            try:
                events.append(message_queue.get_nowait())
            except queue.Empty:
                break
        logs_by_thread = {}
        results_by_thread = {}
        passthrough = []
        for event in events:
            event_type = event.get("type")
            thread_id = event.get("thread_id")
            if event_type == "logs":
                logs_by_thread.setdefault(thread_id, []).append(event.get("response", ""))
            elif event_type == "results":
                results_by_thread[thread_id] = event
            else:
                passthrough.append(event)
        aggregated = []
        for thread_id, chunks in logs_by_thread.items():
            payload = {
                "type": "logs",
                "response": "".join(chunks),
                "thread_id": thread_id,
            }
            aggregated.append(payload)
        aggregated.extend(results_by_thread.values())
        aggregated.extend(passthrough)
        async with clients_lock:
            targets = list(clients)
        if not targets:
            continue
        stale_clients = []
        for event in aggregated:
            payload = json.dumps(event)
            event_type = event.get("type")
            for client in targets:
                if client.is_closed():
                    stale_clients.append(client)
                    continue
                if not client.matches(event):
                    continue
                client.try_enqueue(payload, event_type)
        if stale_clients:
            async with clients_lock:
                for client in stale_clients:
                    clients.discard(client)

async def handle_connection(websocket):
    logger.info("Client connected")
    ensure_observer_started()
    client = ClientState(websocket)
    async with clients_lock:
        clients.add(client)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.warning("Invalid websocket payload: %s", message)
                continue
            if data.get("type") == "subscribe":
                if data.get("all"):
                    client.subscriptions = {"*"}
                    client.subscribed = True
                    continue
                thread_ids = _normalize_thread_ids(
                    data.get("thread_ids")
                    if data.get("thread_ids") is not None
                    else data.get("thread_id")
                )
                client.subscriptions = set(thread_ids)
                client.subscribed = True
            elif data.get("type") == "unsubscribe":
                client.subscriptions = set()
                client.subscribed = True
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        await client.close()
        async with clients_lock:
            clients.discard(client)
        logger.info("Client disconnected")


# --------------------------------------------------
# Server Entrypoint
# --------------------------------------------------

async def main():
    logger.info("WebSocket server running on ws://0.0.0.0:8090")
    ping_interval = _get_env_seconds("AGENT_WS_PING_INTERVAL", 20)
    ping_timeout = _get_env_seconds("AGENT_WS_PING_TIMEOUT", 20)
    close_timeout = _get_env_seconds("AGENT_WS_CLOSE_TIMEOUT", 10)
    compression = "deflate" if AGENT_WS_COMPRESSION else None
    async with websockets.serve(
        handle_connection,
        "0.0.0.0",
        8090,
        ping_interval=ping_interval,
        ping_timeout=ping_timeout,
        close_timeout=close_timeout,
        max_size=AGENT_WS_MAX_SIZE,
        compression=compression,
    ):
        asyncio.create_task(broadcast_loop())
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown")
