import hashlib
import logging
import secrets
from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException

from agent.config import Settings
from agent.factory import build_registry
from agent.orchestrator import OpenAIPlanner, Orchestrator
from agent.schemas import ActionRequest, ChatRequest, ConfirmationRequest

logging.basicConfig(level=logging.INFO, format="%(message)s")
app = FastAPI(title="MLOps Agent", version="1.0.0")


@lru_cache(maxsize=1)
def runtime():
    settings = Settings()
    registry = build_registry(settings)
    return settings, registry, Orchestrator(registry, OpenAIPlanner(settings), settings)


def authenticate(x_api_key: str = Header(default="")) -> str:
    settings = runtime()[0]
    expected = settings.agent_api_key.get_secret_value()
    if not expected or not secrets.compare_digest(expected, x_api_key):
        raise HTTPException(401, "Valid X-API-Key required")
    return hashlib.sha256(x_api_key.encode()).hexdigest()


@app.get("/health")
def health():
    return {"status": "alive"}


@app.post("/chat")
def chat(request: ChatRequest, owner: str = Depends(authenticate)):
    return runtime()[2].chat(request.message, owner)


@app.post("/actions/propose")
def propose(request: ActionRequest, owner: str = Depends(authenticate)):
    try:
        return runtime()[1].propose(request.tool, request.arguments, owner)
    except (ValueError, TypeError):
        raise HTTPException(422, "Invalid or unavailable action") from None


@app.post("/actions/confirm")
def confirm(request: ConfirmationRequest, owner: str = Depends(authenticate)):
    try:
        return runtime()[1].confirm(request.token, owner, request.confirm, uuid4().hex)
    except PermissionError:
        raise HTTPException(403, "Invalid, expired, or already used confirmation") from None
