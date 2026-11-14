import hashlib
import json
import logging
import secrets
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import httpx
from pydantic import BaseModel, ValidationError

from agent.schemas import Evidence

LOG = logging.getLogger("mlops.audit")


@dataclass
class Tool:
    name: str
    description: str
    schema: type[BaseModel]
    handler: Callable
    mutating: bool = False

    def definition(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema.model_json_schema(),
            },
        }


class Registry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self.pending: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.secrets: list[str] = []

    def add(self, tool: Tool):
        if tool.name in self.tools:
            raise ValueError("Duplicate tool")
        self.tools[tool.name] = tool

    def scrub(self, value):
        if isinstance(value, str):
            for secret in self.secrets:
                if len(secret) >= 4:
                    value = value.replace(secret, "[REDACTED]")
            return value
        if isinstance(value, list):
            return [self.scrub(v) for v in value]
        if isinstance(value, dict):
            return {
                k: "[REDACTED]"
                if any(
                    s in k.lower()
                    for s in ("password", "token", "secret", "api_key", "authorization")
                )
                else self.scrub(v)
                for k, v in value.items()
            }
        return value

    def definitions(self) -> list[dict]:
        return [tool.definition() for tool in self.tools.values()]

    def propose(self, name: str, arguments: dict, owner: str) -> dict:
        tool = self.tools.get(name)
        if tool is None or not tool.mutating:
            raise ValueError("Only registered state-changing tools can be proposed")
        validated = tool.schema.model_validate(arguments).model_dump()
        token = secrets.token_urlsafe(32)
        action = {
            "tool": name,
            "arguments": validated,
            "owner": owner,
            "expires": time.monotonic() + 300,
        }
        with self.lock:
            now = time.monotonic()
            self.pending = {
                key: value for key, value in self.pending.items() if value["expires"] >= now
            }
            if len(self.pending) >= 512:
                raise ValueError("Too many pending actions")
            self.pending[hashlib.sha256(token.encode()).hexdigest()] = action
        return {
            "confirmation_token": token,
            "tool": name,
            "arguments": validated,
            "expires_in_seconds": 300,
            "instruction": "Review this exact action and explicitly confirm within five minutes.",
        }

    def confirm(self, token: str, owner: str, confirmed: bool, request_id: str) -> Evidence:
        key = hashlib.sha256(token.encode()).hexdigest()
        with self.lock:
            action = self.pending.get(key)
            if action is None or action["owner"] != owner or action["expires"] < time.monotonic():
                raise PermissionError("Invalid, expired, or already used confirmation")
            self.pending.pop(key)
        if not confirmed:
            return Evidence(
                id="confirmation",
                tool=action["tool"],
                status="denied",
                error="User declined the action",
                request_id=request_id,
            )
        return self.execute(
            action["tool"],
            action["arguments"],
            request_id,
            evidence_id="confirmation",
            authorized=True,
        )

    def execute(
        self,
        name: str,
        arguments: dict,
        request_id: str,
        evidence_id: str = "tool",
        *,
        authorized: bool = False,
    ) -> Evidence:
        start = time.monotonic()
        status, data, error = "ok", {}, None
        try:
            tool = self.tools.get(name)
            if tool is None:
                raise ValueError("Unknown tool")
            values = tool.schema.model_validate(arguments).model_dump()
            if tool.mutating and not authorized:
                raise PermissionError(
                    "Explicit user confirmation is required through the action endpoint"
                )
            data = self.scrub(tool.handler(**values))
            if not isinstance(data, dict):
                raise ValueError("Tool output must be an object")
        except PermissionError as exc:
            status, error = "denied", str(exc)
        except (TimeoutError, httpx.TimeoutException, subprocess.TimeoutExpired):
            status, error = "timeout", "Tool timed out; operational state is unknown."
        except (ValueError, ValidationError):
            status, error = "error", "Invalid tool input or invalid service response."
        except Exception as exc:
            # Never return exception messages: URLs, commands, and headers can contain secrets.
            status, error = (
                "error",
                f"Service unavailable or operation failed ({type(exc).__name__}).",
            )
        if status != "ok":
            data = {}
        LOG.info(
            json.dumps(
                {
                    "request_id": request_id,
                    "tool": name,
                    "status": status,
                    "duration_ms": round((time.monotonic() - start) * 1000, 2),
                }
            )
        )
        return Evidence(
            id=evidence_id, tool=name, status=status, data=data, error=error, request_id=request_id
        )
