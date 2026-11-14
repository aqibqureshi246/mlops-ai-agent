from fastapi.testclient import TestClient

from agent import server
from agent.config import Settings
from agent.registry import Registry, Tool
from agent.schemas import EmptyInput


def test_http_requires_authentication_and_rejects_blank_requests(monkeypatch):
    settings = Settings(agent_api_key="local-test-key")
    monkeypatch.setattr(server, "runtime", lambda: (settings, Registry(), None))
    with TestClient(server.app) as client:
        assert client.post("/chat", json={"message": "hello"}).status_code == 401
        assert (
            client.post(
                "/chat", headers={"X-API-Key": "local-test-key"}, json={"message": "   "}
            ).status_code
            == 422
        )
        assert client.get("/health").status_code == 200


def test_http_confirmation_executes_only_reviewed_action_once(monkeypatch):
    settings = Settings(agent_api_key="local-test-key")
    registry = Registry()
    calls = []
    registry.add(
        Tool("restart", "test", EmptyInput, lambda: calls.append(True) or {"accepted": True}, True)
    )
    monkeypatch.setattr(server, "runtime", lambda: (settings, registry, None))
    headers = {"X-API-Key": "local-test-key"}
    with TestClient(server.app) as client:
        proposed = client.post(
            "/actions/propose", headers=headers, json={"tool": "restart", "arguments": {}}
        )
        assert proposed.status_code == 200 and not calls
        payload = {"token": proposed.json()["confirmation_token"], "confirm": True}
        assert (
            client.post("/actions/confirm", headers=headers, json=payload).json()["status"] == "ok"
        )
        assert client.post("/actions/confirm", headers=headers, json=payload).status_code == 403
        assert len(calls) == 1
