const path = require("path");

// Root of the new Chat3GPP workspace
const root = __dirname;

// Service directories
const ragDir = path.join(root, "rag");
const wsServiceDir = path.join(root, "backend", "ws-service");
const uiDir = path.join(root, "backend", "ui-api");

// Hardcoded conda env for all Python services.
const condaPrefix = "/home/labserver/miniconda3/envs/chat3gpp-clean";
const condaPython = path.join(condaPrefix, "bin", "python");
const condaPathway = path.join(condaPrefix, "bin", "pathway");
const ragPython = condaPython;
const pipelinePython = condaPython;
const uiPython = condaPython;

module.exports = {
  apps: [
    {
      name: "rag-http",
      cwd: ragDir,
      script: ragPython,
      args: ["-m", "uvicorn", "http_serve:app", "--host", "0.0.0.0", "--port", "8000"],
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
    {
      name: "rag-pw-new",
      cwd: ragDir,
      script: "start.py",
      interpreter: ragPython,
      env: {
        VENV_PYTHON: condaPython,
        PATHWAY_BIN: condaPathway,
      },
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
    {
      name: "rag-pw-userkb",
      cwd: ragDir,
      script: "start.py",
      interpreter: ragPython,
      env: {
        VENV_PYTHON: condaPython,
        PATHWAY_BIN: condaPathway,
        PW_SCRIPT: "pw_userkb.py",
      },
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
    {
      name: "rag-server",
      cwd: ragDir,
      script: "rag_server.py",
      interpreter: ragPython,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
    {
      name: "pipeline-ws",
      cwd: wsServiceDir,
      script: "main.py",
      interpreter: pipelinePython,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
    {
      name: "agent-ws",
      cwd: wsServiceDir,
      script: "change.py",
      interpreter: pipelinePython,
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
    {
      name: "ui-flask",
      cwd: uiDir,
      script: uiPython,
      args: ["-m", "gunicorn", "-w", "2", "-b", "0.0.0.0:5001", "app:app"],
      interpreter: "none",
      autorestart: true,
      max_restarts: 10,
      restart_delay: 2000,
    },
  ],
};
