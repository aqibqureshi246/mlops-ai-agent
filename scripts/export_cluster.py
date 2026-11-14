"""Export and validate actual Kubernetes deployment observations."""

import argparse
import json
import time
from pathlib import Path
from uuid import uuid4

import httpx

from agent.config import Settings
from agent.factory import build_registry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    registry = build_registry(Settings())
    request_id = uuid4().hex
    evidence = [
        registry.execute(tool, {}, request_id).model_dump()
        for tool in (
            "kubernetes_deployments",
            "kubernetes_pods",
            "kubernetes_services",
            "kubernetes_events",
        )
    ]
    if any(e["status"] != "ok" for e in evidence):
        raise SystemExit(
            "Cannot export a successful deployment: Kubernetes evidence is unavailable"
        )
    deployments = evidence[0]["data"]["deployments"]
    if not deployments or any(d["desired"] < 2 or d["ready"] != d["desired"] for d in deployments):
        raise SystemExit("Deployment has not reached at least two ready replicas")
    pods = evidence[1]["data"]["pods"]
    if len(pods) < 2 or any(
        not any(c["type"] == "Ready" and c["status"] == "True" for c in p["conditions"])
        for p in pods
    ):
        raise SystemExit("Application Pods are not all Ready")
    version, prediction = None, None
    for _ in range(30):
        try:
            with httpx.Client(base_url=args.api_url, timeout=5) as client:
                response = client.get("/version")
                response.raise_for_status()
                version = response.json()
                result = client.post(
                    "/predict",
                    json={
                        "sepal_length": 5.1,
                        "sepal_width": 3.5,
                        "petal_length": 1.4,
                        "petal_width": 0.2,
                    },
                )
                result.raise_for_status()
                prediction = result.json()
            break
        except httpx.HTTPError:
            time.sleep(2)
    if not version or not prediction or prediction["class_name"] != "setosa":
        raise SystemExit("Real prediction API validation failed")
    if any(d["model_run_id"] != version["model_run_id"] for d in deployments):
        raise SystemExit("Application version disagrees with deployment provenance annotation")
    output = Path("evidence/kubernetes-deployment.json")
    output.write_text(
        json.dumps(
            {
                "request_id": request_id,
                "evidence": evidence,
                "api_version": version,
                "prediction": prediction,
            },
            indent=2,
        )
        + "\n"
    )
    print("Exported verified Kubernetes readiness observations.")


if __name__ == "__main__":
    main()
