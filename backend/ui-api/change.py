"""
WebSocket server that watches pipeline files and streams updates to clients.

- output/artifacts/<thread>/ProcessLogs.md → plain appended logs
- output/artifacts/<thread>/Results.csv     → structured table for results panel
"""

import time
import json
import threading
import asyncio
import websockets
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
from pathlib import Path
import csv
import os

DEFAULT_PIPELINE_ROOT = Path(__file__).resolve().parents[2] / "pipeline"
PIPELINE_ROOT = Path(os.getenv("PIPELINE_ROOT", DEFAULT_PIPELINE_ROOT)).resolve()
ARTIFACTS_DIR = Path(
    os.getenv("PIPELINE_ARTIFACTS_DIR") or (PIPELINE_ROOT / "output" / "artifacts")
).resolve()
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


class MyHandler(FileSystemEventHandler):
    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue
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

    def handle_logs(self, path: Path, thread_id):
        try:
            offset = self.log_offsets.get(path, 0)
            if path.stat().st_size < offset:
                offset = 0

            with open(path, "r") as f:
                f.seek(offset)
                new_content = f.read()
                self.log_offsets[path] = f.tell()

            if new_content.strip():
                self.message_queue.put({
                    "type": "logs",
                    "response": new_content,
                    "thread_id": thread_id
                })
        except FileNotFoundError:
            pass

    def handle_results(self, path: Path, thread_id):
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.message_queue.put({
                "type": "results",
                "format": "table",
                "columns": reader.fieldnames,
                "rows": rows,
                "thread_id": thread_id
            })
        except FileNotFoundError:
            pass


def start_observer(message_queue):
    event_handler = MyHandler(message_queue)
    observer = Observer()
    observer.schedule(event_handler, path=str(ARTIFACTS_DIR), recursive=True)
    observer.start()

    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


async def handle_connection(websocket):
    print("Client connected")
    message_queue = queue.Queue()

    observer_thread = threading.Thread(
        target=start_observer,
        args=(message_queue,),
        daemon=True
    )
    observer_thread.start()

    try:
        while True:
            try:
                event_data = message_queue.get_nowait()
                await websocket.send(json.dumps(event_data))
            except queue.Empty:
                await asyncio.sleep(0.2)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")


async def main():
    print("WebSocket server running on ws://0.0.0.0:8090")
    async with websockets.serve(handle_connection, "0.0.0.0", 8090):
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer shutdown")
