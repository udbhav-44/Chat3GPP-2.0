"""
This module provides an HTTP server for file uploads using FastAPI.
"""
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os
import time
import jwt
import re
import logging
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
logging.basicConfig(level=logging.INFO)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

UPLOAD_DIR = "user_uploads"  # Directory to store uploaded files
CHAT_UPLOAD_DIR = os.getenv("CHAT_UPLOAD_DIR", "chat_uploads")
CHAT_IMAGE_MAX_BYTES = int(os.getenv("CHAT_IMAGE_MAX_BYTES", str(10 * 1024 * 1024)))
CHAT_IMAGE_ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp"}
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHAT_UPLOAD_DIR, exist_ok=True)
UPLOAD_ROOT = Path(UPLOAD_DIR).resolve()
CHAT_UPLOAD_ROOT = Path(CHAT_UPLOAD_DIR).resolve()
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
REQUIRE_AUTH = os.getenv("AUTH_REQUIRE_TOKEN", "false").lower() == "true"

def _sanitize_user_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", value.strip())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid user id.")
    return cleaned


def _get_user_id(request: Request) -> str:
    auth_header = request.headers.get("Authorization", "")
    token = None
    if auth_header.startswith("Bearer "):
        token = auth_header.replace("Bearer ", "", 1).strip()
    if not token:
        token = request.query_params.get("token")
    if not token:
        if REQUIRE_AUTH:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        return "anonymous"
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
    except Exception:
        if REQUIRE_AUTH:
            raise HTTPException(status_code=401, detail="Unauthorized.")
        return "anonymous"
    user_id = payload.get("sub") or payload.get("email")
    if not user_id:
        raise HTTPException(status_code=401, detail="Unauthorized.")
    return _sanitize_user_id(str(user_id))


def _user_root(request: Request) -> Path:
    user_id = _get_user_id(request)
    user_dir = (UPLOAD_ROOT / user_id).resolve()
    if UPLOAD_ROOT not in user_dir.parents and user_dir != UPLOAD_ROOT:
        raise HTTPException(status_code=400, detail="Invalid user directory.")
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def _sanitize_thread_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]", "_", value.strip())
    if not cleaned:
        raise HTTPException(status_code=400, detail="Invalid thread id.")
    return cleaned


def _thread_root(request: Request, thread_id: str) -> Path:
    user_id = _get_user_id(request)
    thread_id = _sanitize_thread_id(thread_id)
    user_dir = (CHAT_UPLOAD_ROOT / user_id / thread_id).resolve()
    if CHAT_UPLOAD_ROOT not in user_dir.parents and user_dir != CHAT_UPLOAD_ROOT:
        raise HTTPException(status_code=400, detail="Invalid thread directory.")
    user_dir.mkdir(parents=True, exist_ok=True)
    return user_dir


def _validate_image_file(filename: str, content_type: str | None, size_bytes: int):
    ext = Path(filename).suffix.lower()
    if ext not in CHAT_IMAGE_ALLOWED_EXT:
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    if content_type and not content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Unsupported image type.")
    if size_bytes > CHAT_IMAGE_MAX_BYTES:
        raise HTTPException(status_code=400, detail="Image exceeds size limit.")


def _safe_upload_path(root: Path, filename: str) -> Path:
    safe_name = Path(filename).name
    target = (root / safe_name).resolve()
    if root not in target.parents and target != root:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    return target


def _file_payload(path: Path, owner: str) -> dict:
    stats = path.stat()
    return {
        "name": path.name,
        "size": stats.st_size,
        "modified_at": datetime.fromtimestamp(stats.st_mtime, tz=timezone.utc).isoformat(),
        "owner": owner,
    }


@app.get("/uploads")
async def list_uploads(request: Request):
    files = []
    user_root = _user_root(request)
    if not user_root.exists():
        return {"files": files}
    owner = user_root.name
    for entry in user_root.iterdir():
        if entry.is_file():
            files.append(_file_payload(entry, owner))
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"files": files}


@app.get("/chat-uploads")
async def list_chat_uploads(request: Request, thread_id: str):
    files = []
    thread_root = _thread_root(request, thread_id)
    if not thread_root.exists():
        return {"files": files}
    owner = thread_root.parent.name
    for entry in thread_root.iterdir():
        if entry.is_file():
            files.append(_file_payload(entry, owner))
    files.sort(key=lambda item: item["modified_at"], reverse=True)
    return {"files": files}


@app.post("/chat-upload")
async def upload_chat_image(request: Request, file: UploadFile = File(...), thread_id: str = ""):
    if not thread_id:
        raise HTTPException(status_code=400, detail="thread_id is required.")
    try:
        original_filename = file.filename
        if not original_filename:
            raise HTTPException(status_code=400, detail="Invalid filename.")
        thread_root = _thread_root(request, thread_id)
        file_path = _safe_upload_path(thread_root, original_filename)
        if file_path.exists():
            timestamp = int(time.time())
            name, extension = os.path.splitext(original_filename)
            new_filename = f"{name}_{timestamp}{extension}"
            file_path = _safe_upload_path(thread_root, new_filename)
        else:
            new_filename = original_filename

        content = await file.read()
        _validate_image_file(new_filename, file.content_type, len(content))
        with open(file_path, "wb") as buffer:
            buffer.write(content)

        return {
            "message": "Image uploaded successfully",
            "filename": new_filename,
            "size": len(content),
            "content_type": file.content_type,
            "thread_id": thread_id,
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Chat image upload failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/chat-upload/{thread_id}/{filename}")
async def get_chat_image(request: Request, thread_id: str, filename: str):
    thread_root = _thread_root(request, thread_id)
    target = _safe_upload_path(thread_root, filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(target)


@app.delete("/chat-upload/{thread_id}/{filename}")
async def delete_chat_image(request: Request, thread_id: str, filename: str):
    thread_root = _thread_root(request, thread_id)
    target = _safe_upload_path(thread_root, filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    target.unlink()
    return {"message": "File deleted successfully", "filename": target.name}

@app.post("/upload")
async def upload_file(request: Request, file: UploadFile = File(...), filename: str = Form(None)):
    """
    Uploads a file to the server.

    Args:
        file (UploadFile): The file to be uploaded.
        filename (str): The name of the file.

    Returns:
        dict: A dictionary containing a message and the filename.
    """
    try:
        # Use the provided filename if available, otherwise use the original filename
        original_filename = filename or file.filename
        
        # Check if a file with the same name already exists
        user_root = _user_root(request)
        file_path = _safe_upload_path(user_root, original_filename)
        if file_path.exists():
            # If it exists, add a timestamp to make it unique
            timestamp = int(time.time())
            name, extension = os.path.splitext(original_filename)
            new_filename = f"{name}_{timestamp}{extension}"
            file_path = _safe_upload_path(user_root, new_filename)
        else:
            new_filename = original_filename

        # Save the file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        return {"message": "File uploaded successfully", "filename": new_filename}
    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.delete("/upload/{filename}")
async def delete_upload(request: Request, filename: str):
    user_root = _user_root(request)
    target = _safe_upload_path(user_root, filename)
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found.")
    target.unlink()
    return {"message": "File deleted successfully", "filename": target.name}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
