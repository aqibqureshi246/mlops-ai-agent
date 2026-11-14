"""Capture five real LLM/tool conversations; fail on missing evidence."""

import json
from pathlib import Path

from agent.config import Settings
from agent.factory import build_registry
from agent.orchestrator import OpenAIPlanner, Orchestrator

ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = Settings()
    registry = build_registry(settings)
    orchestrator = Orchestrator(registry, OpenAIPlanner(settings), settings)
    queries = [
        (
            "01-dataset-provenance",
            "Which datasets are tracked by DVC? Retrieve their current hash, Git commit, and changed-data status.",
            {"dvc_dataset_version", "dvc_data_status"},
        ),
        (
            "02-model-selection",
            "Compare the last five training runs and recommend the best measured candidate based on f1_macro. Retrieve its metrics, parameters and artifacts. Explain the limits of this selection.",
            {"mlflow_best_run", "mlflow_run_details"},
        ),
        (
            "03-pipeline-status",
            "Inspect recent iris_training Airflow DAG runs and retrieve task statuses for the latest run. Do not trigger anything.",
            {"airflow_dag_runs", "airflow_task_instances"},
        ),
        (
            "04-deployment-diagnosis",
            "Check whether the prediction API is available in mlops. Inspect pods, deployment, services and events. Explain any unhealthy pods and restart events using evidence.",
            {
                "kubernetes_pods",
                "kubernetes_deployments",
                "kubernetes_services",
                "kubernetes_events",
            },
        ),
        (
            "05-multi-tool-reasoning",
            "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary. Inspect recent Airflow training workflows too. Do not initiate retraining.",
            {
                "mlflow_best_run",
                "mlflow_run_details",
                "dvc_reproduction_steps",
                "kubernetes_deployments",
                "kubernetes_pods",
                "airflow_dag_runs",
            },
        ),
    ]
    directory = ROOT / "evidence/conversations"
    directory.mkdir(parents=True, exist_ok=True)
    failed = []
    for name, query, required in queries:
        response = orchestrator.chat(query)
        successful = {e.tool for e in response.evidence if e.status == "ok"}
        missing = sorted(required - successful)
        output = {
            "query": query,
            "complete": not missing,
            "missing_required_evidence": missing,
            "response": response.model_dump(),
        }
        output["response"]["pending_actions"] = [
            {"tool": a["tool"], "arguments": a["arguments"]} for a in response.pending_actions
        ]
        (directory / f"{name}.json").write_text(
            json.dumps(output, indent=2) + "\n", encoding="utf-8"
        )
        print(f"{name}: {'complete' if not missing else 'INCOMPLETE'}")
        if missing:
            failed.append(name)
    if failed:
        raise SystemExit("Incomplete live conversations: " + ", ".join(failed))


if __name__ == "__main__":
    main()
