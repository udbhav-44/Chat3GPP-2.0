"""
Centralized LLM configuration for provider/model switching using langchain.
"""
import logging
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ENV_PATH, override=False)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4o-mini"

ROLE_MODEL_ENV = {
    "lats": "LLM_MODEL_LATS",
    "graph": "LLM_MODEL_GRAPH",
    "complex": "LLM_MODEL_COMPLEX",
    "guardrails": "LLM_MODEL_GUARDRAILS",
    "classifier": "LLM_MODEL_CLASSIFIER",
}

ROLE_PROVIDER_ENV = {
    "lats": "LLM_PROVIDER_LATS",
    "graph": "LLM_PROVIDER_GRAPH",
    "complex": "LLM_PROVIDER_COMPLEX",
    "guardrails": "LLM_PROVIDER_GUARDRAILS",
    "classifier": "LLM_PROVIDER_CLASSIFIER",
}

PROVIDER_CONFIG = {
    "openai": {
        "api_key_env": "OPEN_AI_API_KEY_30",
        "base_url_env": "OPENAI_API_BASE",
        "default_base_url": None,
    },
    "deepseek": {
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url_env": "DEEPSEEK_API_BASE",
        "default_base_url": "https://api.deepseek.com",
    },
}

PROVIDER_ALIASES = {
    "open-ai": "openai",
    "deepseek-chat": "deepseek",
}

TEMPERATURE_UNSUPPORTED_PREFIXES = ("gpt-5",)
TOP_P_UNSUPPORTED_PREFIXES = ("gpt-5",)
BLOCKED_MODEL_PREFIXES = ("gpt-5",)


def _canonical_model_name(model: str) -> str:
    base = (model or "").lower()
    if "/" in base:
        base = base.split("/")[-1]
    if ":" in base:
        base = base.split(":")[-1]
    return base


def _get_env_value(key: str) -> str | None:
    value = os.getenv(key)
    return value.strip() if value else None


def _normalize_provider(provider: str | None) -> str | None:
    if not provider:
        return None
    provider = provider.strip().lower()
    return PROVIDER_ALIASES.get(provider, provider)


def _resolve_model(model: str | None, role: str | None) -> str:
    if model:
        return model
    if role:
        role_env = ROLE_MODEL_ENV.get(role.lower())
        if role_env:
            role_model = _get_env_value(role_env)
            if role_model:
                return role_model
    return _get_env_value("LLM_MODEL") or DEFAULT_MODEL


def _resolve_provider(model: str | None, provider: str | None, role: str | None) -> str:
    provider = _normalize_provider(provider)
    if provider:
        return provider
    if role:
        role_env = ROLE_PROVIDER_ENV.get(role.lower())
        if role_env:
            role_provider = _get_env_value(role_env)
            if role_provider:
                return _normalize_provider(role_provider) or role_provider
    env_provider = _get_env_value("LLM_PROVIDER")
    if env_provider:
        return _normalize_provider(env_provider) or env_provider
    if model and model.lower().startswith("deepseek"):
        return "deepseek"
    return "openai"


def _provider_settings(provider: str) -> tuple[str | None, str | None]:
    provider = _normalize_provider(provider) or "openai"
    config = PROVIDER_CONFIG.get(provider)
    if not config:
        logger.warning("Unknown provider '%s', defaulting to openai.", provider)
        config = PROVIDER_CONFIG["openai"]
        provider = "openai"
    api_key = _get_env_value(config["api_key_env"])
    base_url = _get_env_value(config["base_url_env"]) or config["default_base_url"]
    return api_key, base_url


def _normalize_temperature(model: str, temperature: float | None) -> float | None:
    canonical = _canonical_model_name(model)
    for prefix in TEMPERATURE_UNSUPPORTED_PREFIXES:
        if canonical.startswith(prefix) or prefix in canonical:
            return 1.0
    return temperature


def _normalize_top_p(model: str, top_p: float | None) -> float | None:
    if top_p is None:
        return None
    canonical = _canonical_model_name(model)
    for prefix in TOP_P_UNSUPPORTED_PREFIXES:
        if canonical.startswith(prefix) or prefix in canonical:
            return None
    return top_p


def resolve_llm_settings(
    model: str | None = None,
    provider: str | None = None,
    role: str | None = None,
) -> tuple[str, str]:
    resolved_model = _resolve_model(model, role)
    resolved_provider = _resolve_provider(resolved_model, provider, role)
    canonical = _canonical_model_name(resolved_model)
    for prefix in BLOCKED_MODEL_PREFIXES:
        if canonical.startswith(prefix) or prefix in canonical:
            logger.warning("Blocked model '%s' requested; falling back to %s.", resolved_model, DEFAULT_MODEL)
            resolved_model = DEFAULT_MODEL
            break
    if model is None and resolved_provider == "deepseek":
        role_env = ROLE_MODEL_ENV.get(role.lower()) if role else None
        role_model = _get_env_value(role_env) if role_env else None
        if not role_model and not _get_env_value("LLM_MODEL"):
            resolved_model = "deepseek-chat"
    return resolved_provider, resolved_model


def get_llm(
    model: str | None = None,
    temperature: float | None = 0.4,
    top_p: float | None = 0.4,
    provider: str | None = None,
    role: str | None = None,
) -> ChatOpenAI:
    resolved_provider, resolved_model = resolve_llm_settings(
        model=model,
        provider=provider,
        role=role,
    )
    temperature = _normalize_temperature(resolved_model, temperature)
    top_p = _normalize_top_p(resolved_model, top_p)
    api_key, base_url = _provider_settings(resolved_provider)

    llm_kwargs = {
        "model": resolved_model,
        "openai_api_key": api_key,
    }
    if base_url:
        llm_kwargs["openai_api_base"] = base_url
    if temperature is not None:
        llm_kwargs["temperature"] = temperature
    if top_p is not None:
        llm_kwargs["top_p"] = top_p

    return ChatOpenAI(**llm_kwargs)


def get_llm_for_role(role: str, **kwargs: object) -> ChatOpenAI:
    return get_llm(role=role, **kwargs)


# Keep these for backward compatibility during refactor.
GPT4o_mini_LATS = get_llm_for_role("lats", temperature=0.4, top_p=0.4)
GPT4o_mini_GraphGen = get_llm_for_role("graph", temperature=0.2, top_p=0.1)
GPT4o_mini_Complex = get_llm_for_role("complex", temperature=0.6, top_p=0.7)
GPT4o_mini_GuardRails = get_llm_for_role("guardrails", temperature=0.2, top_p=0.1)


_message_histories: dict[str, ChatMessageHistory] = {}


def _get_message_history(session_id: str) -> ChatMessageHistory:
    if session_id not in _message_histories:
        _message_histories[session_id] = ChatMessageHistory()
    return _message_histories[session_id]


_conversation_prompt = ChatPromptTemplate.from_messages(
    [
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)


def _build_conversation_runnable(llm: ChatOpenAI) -> RunnableWithMessageHistory:
    return RunnableWithMessageHistory(
        _conversation_prompt | llm,
        _get_message_history,
        input_messages_key="input",
        history_messages_key="history",
    )


conversation_complex = _build_conversation_runnable(GPT4o_mini_Complex)


def run_conversation_complex(
    prompt: str,
    session_id: str = "default",
    model: str | None = None,
    provider: str | None = None,
    temperature: float | None = None,
    top_p: float | None = None,
) -> str:
    if any(value is not None for value in (model, provider, temperature, top_p)):
        llm = get_llm_for_role(
            "complex",
            model=model,
            provider=provider,
            temperature=temperature,
            top_p=top_p,
        )
        runnable = _build_conversation_runnable(llm)
    else:
        runnable = conversation_complex

    response = runnable.invoke(
        {"input": prompt},
        config={"configurable": {"session_id": session_id}},
    )
    return response.content if hasattr(response, "content") else str(response)


def reload_llms():
    global GPT4o_mini_LATS
    global GPT4o_mini_GraphGen
    global GPT4o_mini_Complex
    global GPT4o_mini_GuardRails
    global conversation_complex

    load_dotenv(ENV_PATH, override=True)

    GPT4o_mini_LATS = get_llm_for_role("lats", temperature=0.4, top_p=0.4)
    GPT4o_mini_GraphGen = get_llm_for_role("graph", temperature=0.2, top_p=0.1)
    GPT4o_mini_Complex = get_llm_for_role("complex", temperature=0.6, top_p=0.7)
    GPT4o_mini_GuardRails = get_llm_for_role("guardrails", temperature=0.2, top_p=0.1)

    conversation_complex = _build_conversation_runnable(GPT4o_mini_Complex)
