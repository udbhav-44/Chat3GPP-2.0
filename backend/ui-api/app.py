from flask import Flask, request, send_file, redirect, jsonify, url_for
from flask_cors import CORS
from authlib.integrations.flask_client import OAuth
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from convert import convert_to_html
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse
import csv
import json
import hashlib
import os
import re
import jwt
import secrets
import smtplib
import sqlite3
from email.message import EmailMessage

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", os.getenv("JWT_SECRET", "change-me"))
_CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", os.getenv("FRONTEND_URL", "http://localhost:6666")).split(",")
    if o.strip()
]
CORS(app, origins=_CORS_ORIGINS, supports_credentials=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.abspath(
    os.getenv("PIPELINE_ROOT", os.path.join(BASE_DIR, "..", "..", "pipeline"))
)
PIPELINE_ARTIFACTS_DIR = os.path.abspath(
    os.getenv("PIPELINE_ARTIFACTS_DIR", os.path.join(PIPELINE_ROOT, "output", "artifacts"))
)
AUTH_DB_PATH = os.getenv("AUTH_DB_PATH", os.path.join(BASE_DIR, "auth.db"))
CHAT_DB_PATH = os.getenv("CHAT_DB_PATH", AUTH_DB_PATH)
JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXP_MINUTES = int(os.getenv("JWT_EXP_MINUTES", "10080"))
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:6666")
RESET_URL_BASE = os.getenv("RESET_URL_BASE", FRONTEND_URL)
AUTH_DEV_MODE = os.getenv("AUTH_DEV_MODE", "false").lower() == "true"
AUTH_TOKEN_IN_URL = os.getenv("AUTH_TOKEN_IN_URL", "true").lower() == "true"
AUTH_REQUIRE_TOKEN = os.getenv("AUTH_REQUIRE_TOKEN", "false").lower() == "true"
REPORTS_DIR = os.getenv("REPORTS_DIR") or BASE_DIR
FEEDBACK_LOG_PATH = os.getenv("FEEDBACK_LOG_PATH", os.path.join(BASE_DIR, "feedback_log.jsonl"))
WRITE_UI_ARTIFACTS = os.getenv("WRITE_UI_ARTIFACTS", os.getenv("WRITE_ARTIFACTS", "false")).lower() in {
    "1",
    "true",
    "yes",
}

SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USER or "no-reply@example.com")
SMTP_TLS = os.getenv("SMTP_TLS", "true").lower() == "true"

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

frontend_host = urlparse(FRONTEND_URL).hostname
use_secure_cookies = FRONTEND_URL.startswith("https")
if frontend_host and frontend_host not in {"localhost", "127.0.0.1"}:
    app.config["SESSION_COOKIE_DOMAIN"] = frontend_host
app.config["SESSION_COOKIE_SAMESITE"] = "None" if use_secure_cookies else "Lax"
app.config["SESSION_COOKIE_SECURE"] = use_secure_cookies

oauth = OAuth(app)
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        api_base_url="https://openidconnect.googleapis.com/v1/",
        client_kwargs={"scope": "openid email profile"},
    )

# with open('companies.json', 'w') as file:
#     json.dump([], file)


def init_auth_db():
    with sqlite3.connect(AUTH_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT,
                provider TEXT NOT NULL DEFAULT 'local',
                google_sub TEXT,
                created_at TEXT NOT NULL,
                reset_token_hash TEXT,
                reset_token_expires_at INTEGER
            )
            """
        )
        conn.commit()


def init_chat_db():
    with sqlite3.connect(CHAT_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_threads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT,
                last_message TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                response_id TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_message_attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                message_id INTEGER NOT NULL,
                thread_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                mime_type TEXT,
                size INTEGER,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def get_db_connection():
    conn = sqlite3.connect(AUTH_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_chat_db_connection():
    conn = sqlite3.connect(CHAT_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def serialize_user(row):
    if not row:
        return None
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "provider": row["provider"],
    }


def create_access_token(user_row):
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_row["id"]),
        "email": user_row["email"],
        "name": user_row["name"] or "",
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=JWT_EXP_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token):
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])


def get_bearer_token():
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        return auth_header.replace("Bearer ", "", 1).strip()
    return None


def get_request_token():
    return get_bearer_token() or request.cookies.get("auth_token")


def require_auth():
    token = get_request_token()
    if not token:
        return False
    try:
        decode_access_token(token)
        return True
    except jwt.InvalidTokenError:
        return False


def get_user_from_token():
    token = get_request_token()
    if not token:
        return None
    try:
        payload = decode_access_token(token)
    except jwt.InvalidTokenError:
        return None
    return {
        "id": payload.get("sub"),
        "email": payload.get("email"),
        "name": payload.get("name"),
    }


def send_reset_email(recipient, reset_url):
    if not SMTP_HOST or not SMTP_USER or not SMTP_PASS:
        return False, "SMTP is not configured"
    message = EmailMessage()
    message["Subject"] = "Reset your UI-3GPP password"
    message["From"] = SMTP_FROM
    message["To"] = recipient
    message.set_content(
        "\n".join(
            [
                "We received a request to reset your password.",
                f"Reset it here: {reset_url}",
                "If you did not request this, you can ignore this email.",
            ]
        )
    )
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            if SMTP_TLS:
                server.starttls()
            server.login(SMTP_USER, SMTP_PASS)
            server.send_message(message)
        return True, None
    except Exception as exc:
        return False, str(exc)


init_auth_db()
init_chat_db()


def normalize_email(email):
    return email.strip().lower()


def get_user_by_email(email):
    with get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE email = ?",
            (email,),
        ).fetchone()


def get_user_by_id(user_id):
    with get_db_connection() as conn:
        return conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()


def _sanitize_thread_id(thread_id):
    if thread_id is None:
        return None
    value = str(thread_id).strip()
    if not value:
        return None
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def _load_results_table(thread_id):
    safe_thread_id = _sanitize_thread_id(thread_id)
    if not safe_thread_id:
        return {"columns": [], "rows": [], "updated_at": None}
    csv_path = os.path.join(PIPELINE_ARTIFACTS_DIR, safe_thread_id, "Results.csv")
    if not os.path.exists(csv_path):
        return {"columns": [], "rows": [], "updated_at": None}
    try:
        with open(csv_path, "r", newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            columns = reader.fieldnames or []
        updated_at = int(os.path.getmtime(csv_path) * 1000)
        return {"columns": columns, "rows": rows, "updated_at": updated_at}
    except FileNotFoundError:
        return {"columns": [], "rows": [], "updated_at": None}


def require_user_id():
    user = get_user_from_token()
    if not user or not user.get("id"):
        return None
    try:
        return int(user["id"])
    except (TypeError, ValueError):
        return None


def update_user_google(user_id, google_sub):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET google_sub = ?, provider = ? WHERE id = ?",
            (google_sub, "google", user_id),
        )
        conn.commit()


def store_reset_token(user_id, token_hash, expires_at):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET reset_token_hash = ?, reset_token_expires_at = ? WHERE id = ?",
            (token_hash, expires_at, user_id),
        )
        conn.commit()


def clear_reset_token(user_id):
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET reset_token_hash = NULL, reset_token_expires_at = NULL WHERE id = ?",
            (user_id,),
        )
        conn.commit()


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({"status": "ok"}), 200


@app.route("/api/auth/signup", methods=["POST"])
def auth_signup():
    data = request.get_json() or {}
    name = data.get("name", "").strip()
    email = normalize_email(data.get("email", ""))
    password = data.get("password", "")

    if not email or "@" not in email:
        return jsonify({"error": "Valid email is required"}), 400
    if not password or len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    if get_user_by_email(email):
        return jsonify({"error": "Account already exists"}), 409

    password_hash = generate_password_hash(password)
    created_at = datetime.now(timezone.utc).isoformat()

    with get_db_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO users (name, email, password_hash, provider, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (name, email, password_hash, "local", created_at),
        )
        conn.commit()
        user_id = cursor.lastrowid

    user_row = get_user_by_id(user_id)
    token = create_access_token(user_row)
    return jsonify({"token": token, "user": serialize_user(user_row)}), 201


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    email = normalize_email(data.get("email", ""))
    password = data.get("password", "")

    user = get_user_by_email(email)
    if not user or not user["password_hash"]:
        return jsonify({"error": "Invalid email or password"}), 401

    if not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    token = create_access_token(user)
    return jsonify({"token": token, "user": serialize_user(user)}), 200


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    token = get_bearer_token()
    if not token:
        return jsonify({"error": "Missing token"}), 401
    try:
        payload = decode_access_token(token)
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired"}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token"}), 401

    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid token"}), 401

    user = get_user_by_id(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404

    return jsonify({"user": serialize_user(user)}), 200


@app.route("/api/auth/forgot", methods=["POST"])
def auth_forgot():
    data = request.get_json() or {}
    email = normalize_email(data.get("email", ""))

    user = get_user_by_email(email)
    if not user:
        return jsonify({"message": "If the account exists, a reset link has been sent."}), 200

    raw_token = secrets.token_urlsafe(32)
    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    expires_at = int((datetime.now(timezone.utc) + timedelta(hours=1)).timestamp())
    store_reset_token(user["id"], token_hash, expires_at)

    reset_url = f"{RESET_URL_BASE}?reset_token={raw_token}&email={email}"
    sent, error = send_reset_email(email, reset_url)

    if sent:
        return jsonify({"message": "If the account exists, a reset link has been sent."}), 200

    if AUTH_DEV_MODE:
        return jsonify({"message": "Reset link generated.", "reset_url": reset_url}), 200

    return jsonify({"error": error or "Failed to send reset email"}), 500


@app.route("/api/auth/reset", methods=["POST"])
def auth_reset():
    data = request.get_json() or {}
    email = normalize_email(data.get("email", ""))
    raw_token = data.get("token", "")
    password = data.get("password", "")

    if not password or len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    user = get_user_by_email(email)
    if not user or not user["reset_token_hash"]:
        return jsonify({"error": "Invalid reset token"}), 400

    token_hash = hashlib.sha256(raw_token.encode("utf-8")).hexdigest()
    if token_hash != user["reset_token_hash"]:
        return jsonify({"error": "Invalid reset token"}), 400

    expires_at = user["reset_token_expires_at"] or 0
    if int(expires_at) < int(datetime.now(timezone.utc).timestamp()):
        return jsonify({"error": "Reset token expired"}), 400

    new_hash = generate_password_hash(password)
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ? WHERE id = ?",
            (new_hash, user["id"]),
        )
        conn.commit()
    clear_reset_token(user["id"])

    return jsonify({"message": "Password updated"}), 200


@app.route("/api/auth/google/login", methods=["GET"])
def auth_google_login():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({"error": "Google OAuth is not configured"}), 500
    redirect_uri = GOOGLE_REDIRECT_URI or url_for("auth_google_callback", _external=True)
    return oauth.google.authorize_redirect(redirect_uri, prompt="select_account")


@app.route("/api/auth/google/callback", methods=["GET"])
def auth_google_callback():
    if not GOOGLE_CLIENT_ID or not GOOGLE_CLIENT_SECRET:
        return jsonify({"error": "Google OAuth is not configured"}), 500
    try:
        token = oauth.google.authorize_access_token()
        try:
            user_info = oauth.google.get("userinfo").json()
        except Exception:
            userinfo_endpoint = (
                oauth.google.server_metadata.get("userinfo_endpoint")
                if getattr(oauth.google, "server_metadata", None)
                else None
            )
            userinfo_endpoint = userinfo_endpoint or "https://openidconnect.googleapis.com/v1/userinfo"
            user_info = oauth.google.get(userinfo_endpoint).json()
    except Exception:
        app.logger.exception(
            "Google OAuth callback failed (error=%s, description=%s)",
            request.args.get("error"),
            request.args.get("error_description"),
        )
        return redirect(f"{FRONTEND_URL}?oauth_error=google")

    email = normalize_email(user_info.get("email", ""))
    if not email:
        return redirect(f"{FRONTEND_URL}?oauth_error=google")

    name = user_info.get("name", "")
    google_sub = user_info.get("sub")
    user = get_user_by_email(email)

    if user:
        if google_sub and user["google_sub"] != google_sub:
            update_user_google(user["id"], google_sub)
        user = get_user_by_id(user["id"])
    else:
        created_at = datetime.now(timezone.utc).isoformat()
        with get_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users (name, email, provider, google_sub, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, email, "google", google_sub, created_at),
            )
            conn.commit()
            user = get_user_by_id(cursor.lastrowid)

    jwt_token = create_access_token(user)
    if AUTH_TOKEN_IN_URL:
        redirect_url = f"{FRONTEND_URL}?token={jwt_token}&provider=google"
        return redirect(redirect_url)

    response = redirect(FRONTEND_URL)
    response.set_cookie(
        "auth_token",
        jwt_token,
        httponly=True,
        secure=FRONTEND_URL.startswith("https"),
        samesite="Lax",
    )
    return response

@app.route("/api/chats", methods=["GET", "POST"])
def chats_collection():
    user_id = require_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == "POST":
        data = request.get_json() or {}
        title = (data.get("title") or "").strip() or "New chat"
        now = datetime.now(timezone.utc).isoformat()
        with get_chat_db_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO chat_threads (user_id, title, last_message, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, title, "", now, now),
            )
            conn.commit()
            thread_id = cursor.lastrowid

        return jsonify({"id": thread_id, "title": title, "created_at": now, "updated_at": now}), 201

    with get_chat_db_connection() as conn:
        rows = conn.execute(
            """
            SELECT id, title, last_message, created_at, updated_at
            FROM chat_threads
            WHERE user_id = ?
            ORDER BY updated_at DESC
            """,
            (user_id,),
        ).fetchall()

    threads = [
        {
            "id": row["id"],
            "title": row["title"],
            "last_message": row["last_message"] or "",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }
        for row in rows
    ]
    return jsonify({"threads": threads}), 200


@app.route("/api/chats/<int:thread_id>", methods=["GET", "DELETE"])
def chat_detail(thread_id):
    user_id = require_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    if request.method == "DELETE":
        with get_chat_db_connection() as conn:
            thread = conn.execute(
                "SELECT id FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            ).fetchone()
            if not thread:
                return jsonify({"error": "Chat not found"}), 404

            conn.execute(
                "DELETE FROM chat_messages WHERE thread_id = ?",
                (thread_id,),
            )
            conn.execute(
                "DELETE FROM chat_threads WHERE id = ? AND user_id = ?",
                (thread_id, user_id),
            )
            conn.commit()

        return jsonify({"deleted": True, "id": thread_id}), 200

    with get_chat_db_connection() as conn:
        thread = conn.execute(
            """
            SELECT id, title, last_message, created_at, updated_at
            FROM chat_threads
            WHERE id = ? AND user_id = ?
            """,
            (thread_id, user_id),
        ).fetchone()

        if not thread:
            return jsonify({"error": "Chat not found"}), 404

        messages = conn.execute(
            """
            SELECT id, role, content, source, response_id, created_at
            FROM chat_messages
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()

        attachments = conn.execute(
            """
            SELECT id, message_id, filename, mime_type, size, created_at
            FROM chat_message_attachments
            WHERE thread_id = ?
            ORDER BY id ASC
            """,
            (thread_id,),
        ).fetchall()

    attachments_by_message = {}
    for row in attachments:
        attachments_by_message.setdefault(row["message_id"], []).append(
            {
                "id": row["id"],
                "filename": row["filename"],
                "mime_type": row["mime_type"],
                "size": row["size"],
                "created_at": row["created_at"],
            }
        )

    return jsonify(
        {
            "thread": {
                "id": thread["id"],
                "title": thread["title"],
                "last_message": thread["last_message"] or "",
                "created_at": thread["created_at"],
                "updated_at": thread["updated_at"],
            },
            "messages": [
                {
                    "id": row["id"],
                    "role": row["role"],
                    "content": row["content"],
                    "source": row["source"],
                    "response_id": row["response_id"],
                    "created_at": row["created_at"],
                    "attachments": attachments_by_message.get(row["id"], []),
                }
                for row in messages
            ],
        }
    ), 200


@app.route("/api/chats/<int:thread_id>/results", methods=["GET"])
def chat_results(thread_id):
    user_id = require_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    with get_chat_db_connection() as conn:
        thread = conn.execute(
            "SELECT id FROM chat_threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        ).fetchone()

    if not thread:
        return jsonify({"error": "Chat not found"}), 404

    return jsonify(_load_results_table(thread_id)), 200


@app.route("/api/chats/<int:thread_id>/messages", methods=["POST"])
def chat_messages(thread_id):
    user_id = require_user_id()
    if not user_id:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    role = data.get("role")
    content = (data.get("content") or "").strip()
    attachments = data.get("attachments") or []
    if role not in ("user", "assistant"):
        return jsonify({"error": "Role must be 'user' or 'assistant'"}), 400
    if not content:
        return jsonify({"error": "Content is required"}), 400
    if attachments and not isinstance(attachments, list):
        return jsonify({"error": "Attachments must be a list"}), 400
    if isinstance(attachments, list) and len(attachments) > 6:
        return jsonify({"error": "Too many attachments"}), 400

    with get_chat_db_connection() as conn:
        thread = conn.execute(
            "SELECT id, title FROM chat_threads WHERE id = ? AND user_id = ?",
            (thread_id, user_id),
        ).fetchone()
        if not thread:
            return jsonify({"error": "Chat not found"}), 404

        now = datetime.now(timezone.utc).isoformat()
        title = thread["title"] or ""
        if role == "user" and (not title or title.lower().startswith("new chat")):
            title = content[:80]
            conn.execute(
                "UPDATE chat_threads SET title = ? WHERE id = ?",
                (title, thread_id),
            )
        cursor = conn.execute(
            """
            INSERT INTO chat_messages (thread_id, role, content, source, response_id, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                thread_id,
                role,
                content,
                data.get("source"),
                data.get("response_id"),
                now,
            ),
        )
        conn.execute(
            """
            UPDATE chat_threads
            SET last_message = ?, updated_at = ?
            WHERE id = ?
            """,
            (content[:200], now, thread_id),
        )

        allowed_types = {"image/png", "image/jpeg", "image/webp", "image/jpg"}
        if attachments:
            for attachment in attachments:
                if not isinstance(attachment, dict):
                    continue
                filename = attachment.get("filename")
                if not filename:
                    continue
                mime_type = attachment.get("mime_type")
                if mime_type and mime_type not in allowed_types:
                    continue
                size = attachment.get("size")
                conn.execute(
                    """
                    INSERT INTO chat_message_attachments (message_id, thread_id, filename, mime_type, size, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cursor.lastrowid,
                        thread_id,
                        filename,
                        mime_type,
                        size,
                        now,
                    ),
                )
        conn.commit()

    return jsonify({"id": cursor.lastrowid, "created_at": now}), 201

@app.route('/query', methods=['POST'])
def handle_query():
    # Parse the JSON request
    data = request.json
    query = data.get('query', '')

    # Log or process the received query
    print(f"Received query: {query}")
    return '',204


@app.route('/convert', methods=['POST'])
def convert_to_pdf():
    if AUTH_REQUIRE_TOKEN and not require_auth():
        return {"error": "Unauthorized"}, 401
    if not WRITE_UI_ARTIFACTS:
        return {"error": "Report generation is disabled."}, 503
    data = request.get_json()
    markdown_content = data.get('content', '')
    request_id = _sanitize_thread_id(data.get("request_id"))
    markdown_content = str(markdown_content)
    output_base = f"pathway_{request_id}" if request_id else "pathway"
    os.makedirs(REPORTS_DIR, exist_ok=True)
    output_file = convert_to_html(markdown_content, output_base=output_base, output_dir=REPORTS_DIR)
    if not os.path.exists(output_file):
        return {"error": "Failed to generate PDF"}, 500

    return {"message": "PDF generated successfully", "request_id": request_id}

@app.route('/download-pdf', methods=['GET'])
def download_pdf():
    if AUTH_REQUIRE_TOKEN and not require_auth():
        return {"error": "Unauthorized"}, 401
    if not WRITE_UI_ARTIFACTS:
        return {"error": "Report downloads are disabled."}, 503
    request_id = _sanitize_thread_id(request.args.get("request_id"))
    output_base = f"pathway_{request_id}" if request_id else "pathway"
    output_file = os.path.join(REPORTS_DIR, f"{output_base}.html")

    # Check if the PDF exists
    if not os.path.exists(output_file):
        return {"error": "PDF not found"}, 404

    return send_file(output_file, as_attachment=True, download_name=f"{output_base}.html", mimetype='text/html')


@app.route("/api/feedback", methods=["POST"])
def submit_feedback():
    data = request.get_json() or {}
    rating = data.get("rating")
    if rating not in ("up", "down"):
        return jsonify({"error": "Rating must be 'up' or 'down'"}), 400

    token_user = get_user_from_token()
    request_user = data.get("user") if isinstance(data.get("user"), dict) else {}
    user_payload = token_user or {
        "name": request_user.get("name", ""),
        "email": request_user.get("email", ""),
    }

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "comment": data.get("comment", ""),
        "prompt": data.get("prompt", ""),
        "answer": data.get("answer", ""),
        "source": data.get("source", ""),
        "response_id": data.get("response_id"),
        "user": user_payload,
        "ip": request.remote_addr,
        "user_agent": request.headers.get("User-Agent", ""),
        "authenticated": bool(token_user),
    }

    if not WRITE_UI_ARTIFACTS:
        return jsonify({"status": "skipped"}), 201
    try:
        log_dir = os.path.dirname(FEEDBACK_LOG_PATH)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(entry) + "\n")
    except Exception as exc:
        return jsonify({"error": f"Failed to store feedback: {exc}"}), 500

    return jsonify({"status": "ok"}), 201

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5001, debug=True)  # Run Flask server on port 5001
