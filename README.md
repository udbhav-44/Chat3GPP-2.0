# Chat3GPP

Production-ready assistant for 3GPP standards research. It combines a multi-agent WebSocket backend, RAG retrieval services, a Flask auth/API layer, and a React UI, with data in Neo4j, Postgres, and Pathway stores.

## Architecture (what lives where)
- **frontend/**: React/Vite client. Connects via WebSocket to the agent backend; uploads files to the RAG upload API; reads auth tokens from cookies/headers.
- **backend/ws-service/**: WebSocket entrypoint. Classifies queries, plans tasks, runs tools (web search, RAG, Neo4j), orchestrates with LATS, streams partial graphs/artifacts.
- **pipeline/**: Shared agent/tool library used by ws-service. Holds Agents, Tools, LATS, ingestion helpers (`jina-reader`).
- **rag/**: Retrieval and document services.
  - `http_serve.py`: Upload/list/delete per user.
  - `pw_new.py` / `pw_userkb.py`: Pathway doc stores (global vs per-user).
  - `rag_server.py`: Retrieve → optional rerank → generate with OpenAI/DeepSeek.
- **backend/ui-api/**: Flask service for auth (local + Google OAuth), JWTs, chat/thread storage (SQLite), feedback logging, export/convert endpoints.
- **data/neo4j-backups/**: Holds offline Neo4j backups only (never a live store).
- **docker-compose.yml**: Postgres (pgvector) + Adminer. Extend for other infra if desired.
- **pm2.config.js**: Process manager config for all runtime services in this layout.

## Request flow (end-to-end)
1) Browser loads frontend → opens WebSocket to ws-service (JWT in header/cookie).
2) ws-service classifies the query, plans tasks, then calls tools:
   - RAG calls go to `rag_server` → Pathway stores (`pw_new` for global, `pw_userkb` for per-user uploads).
   - Graph data/metadata fetched from Neo4j.
3) Agents stream intermediate graph (`Graph.json`) and message chunks back over WebSocket.
4) UI API manages user auth, stores chat threads/messages, logs feedback, and serves exports; it shares JWT secrets with ws-service and RAG upload endpoints.

## Runtime topology (production options)
- **Process manager**: PM2 via `pm2.config.js` for simple multi-service bring-up on a single host.
- **Containers**: Wrap each service with its own Dockerfile; orchestrate with docker-compose or Kubernetes. Mount volumes for:
  - Postgres data (compose volume)
  - Neo4j data (external managed instance or mounted volume)
  - Pathway stores: `rag/uploads`, `rag/user_uploads`
  - Artifacts: `backend/ws-service/output/artifacts`
- **Networking**: Put nginx/ingress in front of:
  - WebSocket: `wss://<host>/ws` → ws-service:8080
  - Upload API: `https://<host>/upload` → rag/http_serve:8000
  - UI API: `https://<host>/api` → ui-api:5001 (or proxied)
  - Frontend: serve built `frontend/dist` as static assets

## Scaling guidance
- **Stateless fronts**: ws-service, ui-api, rag_server, http_serve are stateless; scale horizontally behind a load balancer. Maintain sticky sessions only if needed for WebSocket; otherwise share JWT secrets across instances.
- **Stateful components**:
  - Pathway stores: each instance keeps its own embedding index; run one per corpus or back them with shared storage.
  - Neo4j: run as a managed single instance or cluster; never copy live store files.
  - Postgres: single instance from compose or managed service.
- **Concurrency limits**: ws-service enforces per-user and global semaphores; tune via env vars.

## Observability
- **Logs**: ws-service writes `ProcessLogs.md` and artifacts; Python logging configured in `logging_config.py`. RAG services log to stdout; Pathway logs to `rag/logs` if enabled. UI API logs to stdout and `feedback_log.jsonl`.
- **Metrics**: Add Prometheus exporters if needed (not bundled). PM2 provides process health/restarts.
- **Tracing**: Not built-in; instrument LangChain calls or FastAPI/Flask with OpenTelemetry if desired.

## Security hardening
- Enforce HTTPS termination at the edge; set `SESSION_COOKIE_SECURE` and `SameSite=None` in UI API (already conditional on HTTPS).
- Rotate `JWT_SECRET`/`SECRET_KEY`; never commit real secrets.
- Gate upload and WebSocket endpoints with JWT; optionally add IP allowlists/rate limiting at nginx/ingress.
- Store model/API keys in env/secret manager; avoid embedding in frontend.

## Configuration (key env knobs)
- **Backend/ws-service & pipeline**
  - `JWT_SECRET`, `JWT_ALGORITHM`
  - `RAG_GENERATE_URL`, `RAG_RETRIEVE_URL`, `RAG_STATS_URL`
  - `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`
  - `LANGGRAPH_CHECKPOINT_URL` (or `CHECKPOINT_DATABASE_URL`)
  - Model selection: `LLM_PROVIDER`, `LLM_MODEL`, per-role overrides
  - Concurrency: `MAX_CONCURRENT_QUERIES`, `MAX_ACTIVE_TASKS_PER_CONNECTION`
- **RAG**
  - `OPEN_AI_API_KEY_30`, `DEEPSEEK_API_KEY`, `VOYAGE_API_KEY`
  - `RAG_RETRIEVE_TIMEOUT`, `RAG_RERANK_ENABLED`, `RERANK_*`, `CONTEXT_*`
  - `RAG_RETRIEVE_URL`, `RAG_USER_RETRIEVE_URL`, `USER_RETRIEVE_*`
  - Upload paths: `RAG_UPLOADS_DIR`, `RAG_USER_UPLOADS_DIR`
- **UI API**
  - `JWT_SECRET`, `SECRET_KEY`, `JWT_EXP_MINUTES`
  - OAuth: `GOOGLE_CLIENT_ID`, `GOOGLE_CLIENT_SECRET`, `GOOGLE_REDIRECT_URI`
  - SMTP: `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `SMTP_TLS`, `SMTP_FROM`
  - Paths: `PIPELINE_ROOT`, `PIPELINE_ARTIFACTS_DIR`, `AUTH_DB_PATH`, `CHAT_DB_PATH`, `FRONTEND_URL`
- **Frontend**
  - `VITE_API_BASE_URL`, `VITE_WS_BASE_URL`, `VITE_UPLOAD_BASE_URL`, `FRONTEND_URL`

## Data management
- **Neo4j**: Use online backup (`neo4j-admin backup --backup-dir data/neo4j-backups --database neo4j`). Do not copy running store files. Restore with `neo4j-admin database load`. Point services to the live URI via env.
- **Uploads**: Ensure `rag/uploads` and `rag/user_uploads` are on durable storage. Back them up on your schedule.
- **Artifacts & checkpoints**: Persist Postgres volume; optionally ship `output/artifacts` to object storage if needed for audit.

## Build and deploy
1) Install deps:
   - `pip install -r backend/ws-service/requirements.txt`
   - `pip install -r backend/ui-api/requirements.txt`
   - `pip install -r rag/requirements.txt`
   - `npm install --prefix frontend`
2) Set env files for each service (see knobs above).
3) Production start (single host): `pm2 start pm2.config.js`
4) Frontend build: `npm run build --prefix frontend`; serve `frontend/dist` via nginx/ingress.
5) Containers (optional): create service-specific Dockerfiles; wire with `docker-compose.yml` or Kubernetes; mount volumes for Postgres, uploads, artifacts, and (if local) Neo4j.

## Ports (defaults)
- WebSocket backend: 8080
- RAG upload API: 8000
- Pathway stores: 4004 (global), 4006 (user)
- RAG generator: 4005
- UI API (gunicorn): 5001
- Frontend dev (Vite): 5173

## Operational checklist
- HTTPS fronting all public endpoints.
- Secrets in env/secret manager; distinct per environment.
- Backups: Neo4j online backups; Postgres volume snapshots; uploads directory backups.
- Monitoring: PM2 health, log aggregation, and (optionally) Prometheus/OpenTelemetry.
- Scaling: load-balance stateless services; ensure shared JWT secret; use sticky sessions for WebSocket if needed; size Pathway instances for embedding memory.
