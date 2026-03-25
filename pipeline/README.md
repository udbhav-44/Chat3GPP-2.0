# 3GPP Pipeline

This project implements a dynamic agentic pipeline specifically designed for processing and answering queries related to 3GPP standards and telecommunications. It leverages a multi-agent architecture with Large Language Models (LLMs) to handle both simple inquiries and complex analytical tasks, capable of generating detailed reports and conducting deep research using Retrieval-Augmented Generation (RAG).

## Features

- **Dynamic Query Classification**: Automatically distinguishes between "simple" queries (requiring concise answers) and "complex" queries (requiring detailed analysis and reports).
- **Multi-Agent Orchestration**: Utilizes a suite of specialized agents including Planner, RAG, Classifier, and Drafter agents to execute tasks.
- **Retrieval-Augmented Generation (RAG)**: Integrates internal 3GPP document context to provide grounded and accurate technical responses.
- **Language Agent Tree Search (LATS)**: Implements advanced reasoning capabilities for sub-agents to explore multiple solution paths.
- **WebSocket API**: Provides a real-time WebSocket interface for client-server communication, supporting streaming responses and execution graphs.
- **Topical Guardrails**: Includes mechanisms to ensure queries remain within the relevant domain scope.
- **Parallel Execution**: Efficiently manages task dependencies and parallel execution of sub-tasks.

## Architecture

The system is built around a central WebSocket server (`main.py`) that handles incoming connections. The processing flow is as follows:

1.  **Query Reception**: A query is received via WebSocket.
2.  **Guardrails**: The query is checked against topical guardrails.
3.  **Classification**: The `ClassifierAgent` determines if the query is "simple" or "complex".
    -   **Simple Queries**: Handled by the `ConciseAnsAgent` or a simple RAG look-up for immediate response.
    -   **Complex Queries**:
        -   The `PlannerAgent` decomposes the query into a set of sub-tasks.
        -   A task graph is generated and sent to the client.
        -   Sub-tasks are executed by specialized agents (using LATS where necessary).
        -   The `Smack` module manages the execution flow.
        -   The `DrafterAgent` compiles the results into a final cohesive report.
4.  **Response**: The final response (and any generated charts or additional questions) is streamed back to the client.

## Directory Structure

-   `main.py`: The entry point for the backend, setting up the WebSocket server and orchestration logic.
-   `Agents/`: Contains the implementation of various agents.
    -   `PlannerAgent.py`: Decomposes complex user queries.
    -   `ClassifierAgent.py`: Determines query complexity (simple vs. complex).
    -   `RAG_Agent.py`: Handles retrieval from internal documents.
    -   `DrafterAgent.py`: Synthesizes results into final reports.
    -   `Smack.py`: Manages agent execution and task dependencies.
    -   `LATS/`: Implementation of the Language Agent Tree Search algorithm.
-   `Tools/`: Contains tool definitions and configurations.
-   `requirements.txt`: List of Python dependencies.
-   `logging_config.py`: Configuration for system logging.
-   `TopicalGuardrails.py`: Logic for filtering out-of-scope queries.

## Installation

1.  **Clone the repository**:
    Access the codebase in your local environment.

2.  **Install Dependencies**:
    Ensure you have Python installed, then run:
    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**:
    Create a `.env` file in the root directory. You will need to configure API keys for the LLMs and other services. Key variables typically include:
    -   `OPEN_AI_API_KEY_30`
    -   `DEEPSEEK_API_KEY` (required when `LLM_PROVIDER=deepseek`)
    -   `GEMINI_API_KEY_30`
    -   `LLM_PROVIDER` (e.g., `openai` or `deepseek`)
    -   `LLM_MODEL` (e.g., `gpt-4o-mini` or `deepseek-chat`)
    -   `LLM_PROVIDER_<ROLE>` / `LLM_MODEL_<ROLE>` (optional per-role overrides; roles: `LATS`, `GRAPH`, `COMPLEX`, `GUARDRAILS`, `CLASSIFIER`)
    -   `OPENAI_API_BASE` or `DEEPSEEK_API_BASE` (optional for custom endpoints)
    -   `LANGCHAIN_VERBOSE` (optional)
    -   `POSTGRES_USER`, `POSTGRES_PASSWORD`, etc. (if using database checkpoints)

## Usage

To start the backend server:

```bash
python main.py
```

The server will start listening on `ws://0.0.0.0:8080`.

**Client Interaction**:
Clients should connect to the WebSocket endpoint and send JSON messages.
-   **Send Query**: `{"type": "query", "query": "Your question here", "thread_id": "optional-id", "model": "gpt-4o-mini", "provider": "openai", "user_id": "user-identifier"}`. When `user_id` is provided, RAG tools are scoped to that user's private uploads.
-   **Toggle RAG**: `{"type": "toggleRag", "query": true/false}`
-   **Abort**: `{"type": "abort"}`
