from datetime import UTC, datetime
from urllib.parse import quote
from uuid import uuid4

import httpx
from pydantic import Field

from agent.schemas import StrictInput


class DagInput(StrictInput):
    dag_id: str = Field(default="iris_training", pattern=r"^[a-zA-Z0-9_.-]{1,100}$")


class DagRunInput(DagInput):
    run_id: str = Field(min_length=1, max_length=250)


class TaskLogInput(DagRunInput):
    task_id: str = Field(pattern=r"^[a-zA-Z0-9_.-]{1,100}$")
    try_number: int = Field(default=1, ge=1, le=20)


class TriggerInput(DagInput):
    quality_threshold: float = Field(default=0.90, ge=0, le=1)
    inject_failure: str = Field(default="none", pattern=r"^(none|validation|quality)$")


class AirflowTools:
    def __init__(
        self,
        url: str,
        username: str,
        password: str,
        approved_dag: str,
        timeout: float = 20,
        client=None,
    ):
        self.http = client or httpx.Client(base_url=url.rstrip("/"), timeout=timeout)
        self.username, self.password, self.approved_dag = username, password, approved_dag

    def _request(self, method: str, path: str, **kwargs) -> dict:
        token = self.http.post(
            "/auth/token", json={"username": self.username, "password": self.password}
        )
        token.raise_for_status()
        response = self.http.request(
            method,
            f"/api/v2/{path}",
            headers={"Authorization": f"Bearer {token.json()['access_token']}"},
            **kwargs,
        )
        response.raise_for_status()
        return response.json()

    def _dag(self, dag_id: str) -> str:
        if dag_id != self.approved_dag:
            raise PermissionError("DAG is outside the allowlist")
        return f"dags/{quote(dag_id, safe='')}"

    def list_dags(self) -> dict:
        result = self._request("GET", "dags", params={"limit": 100})
        return {
            "dags": [
                {k: d.get(k) for k in ("dag_id", "is_paused", "is_stale")}
                for d in result.get("dags", [])
            ]
        }

    def dag_runs(self, dag_id: str = "iris_training") -> dict:
        result = self._request(
            "GET", f"{self._dag(dag_id)}/dagRuns", params={"limit": 10, "order_by": "-logical_date"}
        )
        return {
            "dag_id": dag_id,
            "runs": [
                {
                    k: r.get(k)
                    for k in ("dag_run_id", "state", "start_date", "end_date", "logical_date")
                }
                for r in result.get("dag_runs", [])
            ],
        }

    def task_instances(self, dag_id: str, run_id: str) -> dict:
        result = self._request(
            "GET", f"{self._dag(dag_id)}/dagRuns/{quote(run_id, safe='')}/taskInstances"
        )
        tasks = [
            {k: t.get(k) for k in ("task_id", "state", "try_number", "start_date", "end_date")}
            for t in result.get("task_instances", [])
        ]
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "tasks": tasks,
            "failed_tasks": [t for t in tasks if t["state"] in ("failed", "upstream_failed")],
        }

    def task_logs(self, dag_id: str, run_id: str, task_id: str, try_number: int = 1) -> dict:
        path = (
            f"{self._dag(dag_id)}/dagRuns/{quote(run_id, safe='')}/taskInstances/"
            f"{quote(task_id, safe='')}/logs/{try_number}"
        )
        return {
            "dag_id": dag_id,
            "run_id": run_id,
            "task_id": task_id,
            "logs": self._request("GET", path, params={"full_content": True}),
        }

    def trigger_training(
        self,
        dag_id: str = "iris_training",
        quality_threshold: float = 0.90,
        inject_failure: str = "none",
    ) -> dict:
        path = self._dag(dag_id)
        result = self._request(
            "POST",
            f"{path}/dagRuns",
            json={
                "dag_run_id": f"manual__agent_{uuid4().hex}",
                "logical_date": datetime.now(UTC).isoformat(),
                "conf": {"quality_threshold": quality_threshold, "inject_failure": inject_failure},
            },
        )
        return {
            "dag_id": dag_id,
            "run_id": result["dag_run_id"],
            "state": result["state"],
            "note": "Submission acknowledged; inspect task statuses until a terminal state.",
        }
