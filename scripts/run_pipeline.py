"""Trigger explicitly authorized training and export observed results."""

import argparse
import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

from agent.config import Settings
from agent.factory import build_registry

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--inject-failure", choices=["none", "validation", "quality"], default="none"
    )
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required to authorize a new training DAG run")
    registry = build_registry(Settings())
    request_id = uuid4().hex
    proposal = registry.propose(
        "airflow_trigger_training", {"inject_failure": args.inject_failure}, "demo"
    )
    trigger = registry.confirm(proposal["confirmation_token"], "demo", True, request_id)
    if trigger.status != "ok":
        raise SystemExit(trigger.model_dump_json(indent=2))
    run_id = trigger.data["run_id"]
    evidence = [trigger.model_dump()]
    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        result = registry.execute(
            "airflow_task_instances", {"dag_id": "iris_training", "run_id": run_id}, request_id
        )
        if result.status != "ok":
            raise SystemExit(result.model_dump_json(indent=2))
        tasks = result.data["tasks"]
        if len(tasks) == 5 and all(
            t["state"] in ("success", "failed", "upstream_failed", "skipped") for t in tasks
        ):
            evidence.append(result.model_dump())
            for task in tasks:
                if task["state"] == "failed":
                    logs = registry.execute(
                        "airflow_task_logs",
                        {
                            "dag_id": "iris_training",
                            "run_id": run_id,
                            "task_id": task["task_id"],
                            "try_number": max(1, task["try_number"]),
                        },
                        request_id,
                    )
                    evidence.append(logs.model_dump())
            output = ROOT / "evidence" / f"pipeline-{args.inject_failure}.json"
            output.parent.mkdir(exist_ok=True)
            output.write_text(
                json.dumps(
                    {"run_id": run_id, "injection": args.inject_failure, "evidence": evidence},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"Observed terminal task states. Exported {output.name}")
            failed = [t for t in tasks if t["state"] != "success"]
            if args.inject_failure == "none" and failed:
                raise SystemExit("Training failed; inspect exported evidence.")
            if args.inject_failure != "none" and not any(t["state"] == "failed" for t in tasks):
                raise SystemExit("Expected failure was not observed.")
            if args.inject_failure == "none":
                bundle = (
                    Path("artifacts")
                    / hashlib.sha256(run_id.encode()).hexdigest()[:24]
                    / "approved"
                )
                if not (ROOT / bundle / "metadata.json").exists():
                    raise SystemExit("Successful tasks have no approved model bundle")
                (ROOT / ".runtime/latest-approved.txt").write_text(str(bundle), encoding="utf-8")
                print(f"Approved bundle: {bundle}")
            return
        time.sleep(3)
    raise SystemExit("Timed out waiting for terminal task statuses; no success claimed.")


if __name__ == "__main__":
    main()
