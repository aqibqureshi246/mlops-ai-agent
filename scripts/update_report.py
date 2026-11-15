"""Append a verification record only after inspecting complete live evidence."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def task_states(observation):
    return {
        task["task_id"]: task["state"]
        for item in observation["evidence"]
        if item["tool"] == "airflow_task_instances" and item["status"] == "ok"
        for task in item["data"]["tasks"]
    }


def verify_pipeline_states(success, validation, quality):
    task_ids = {"validate", "preprocess", "train", "evaluate_mlflow", "quality_gate"}
    recovered = task_states(success)
    rejected_validation = task_states(validation)
    rejected_quality = task_states(quality)
    if recovered != dict.fromkeys(task_ids, "success"):
        raise SystemExit("Recovery training did not finish successfully")
    expected_validation = dict.fromkeys(task_ids, "upstream_failed")
    expected_validation["validate"] = "failed"
    if rejected_validation != expected_validation:
        raise SystemExit("Validation rejection did not block all downstream tasks")
    expected_quality = dict.fromkeys(task_ids, "success")
    expected_quality["quality_gate"] = "failed"
    if rejected_quality != expected_quality:
        raise SystemExit("Quality rejection did not occur at the final gate")


def main():
    tests = read("evidence/automated-tests.json")
    if tests["tests"] < 8 or tests["failures"] or tests["errors"]:
        raise SystemExit("Passing automated test evidence is required")
    success = read("evidence/pipeline-none.json")
    validation = read("evidence/pipeline-validation.json")
    quality = read("evidence/pipeline-quality.json")
    reproduced = read("evidence/dvc-reproduction.json")
    cluster = read("evidence/kubernetes-deployment.json")
    conversations = list(Path("evidence/conversations").glob("*.json"))
    if len(conversations) != 5 or not all(read(p)["complete"] for p in conversations):
        raise SystemExit("Five complete conversations are required before updating the report")
    if not reproduced["matches"] or not cluster["evidence"]:
        raise SystemExit("Dataset or deployment evidence is missing")
    if "Injected validation failure" not in json.dumps(validation):
        raise SystemExit("Validation failure root cause is not present in the exported logs")
    if "Quality gate rejected candidate" not in json.dumps(quality):
        raise SystemExit("Quality failure root cause is not present in the exported logs")
    verify_pipeline_states(success, validation, quality)
    result = {
        "verified_at": datetime.now(UTC).isoformat(),
        "recovery_run": success["run_id"],
        "validation_failure_run": validation["run_id"],
        "quality_failure_run": quality["run_id"],
        "dataset_hash": reproduced["retrieved_hash"],
        "complete_conversations": len(conversations),
        "tests_passed": tests["tests"] - tests.get("skipped", 0),
        "test_python": tests["python"],
        "test_platform": tests["platform"],
    }
    if os.environ.get("GITHUB_RUN_ID") and os.environ.get("GITHUB_REPOSITORY"):
        result["workflow_url"] = (
            f"https://github.com/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
            f"{os.environ['GITHUB_RUN_ID']}"
        )
        result["source_commit"] = os.environ.get("GITHUB_SHA", "unknown")
    Path("evidence/verification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = Path("docs/technical-report.md")
    text = report.read_text()
    header = text.index("## Validation record")
    table = text.index("| Evidence |", header)
    text = (
        text[:header]
        + "## Validation record\n\nThe live workflow verified the observations below. Actual run IDs, source revision and verification time are recorded at the end of this report.\n\n"
        + text[table:]
    )
    text = (
        "\n".join(
            (
                f"| Automated tests | evidence/automated-tests.json | {result['tests_passed']} passed; "
                f"{tests.get('skipped', 0)} skipped; Python {tests['python']} on {tests['platform']} |"
                if line.startswith("| Automated tests |")
                else line
            )
            for line in text.splitlines()
        )
        + "\n"
    )
    text = text.replace("Pending live execution", "Verified; see exported evidence")
    text = text.replace("Pending execution", "Passed; see automated-tests.json")
    text = text.replace(
        "These are planned reproducible investigations until the corresponding real exports exist.",
        "Both failure investigations were executed and their root causes were verified in the exported logs.",
    )
    text = text.replace("Expected diagnosis:", "Verified outcome:")
    text = text.replace(
        "Actual observations and\nrun IDs belong in",
        "Actual observations and\nrun IDs are recorded in",
    )
    text = text.replace("Actual logs belong in", "Actual logs are recorded in")
    text = text.replace(
        "two planned deliberately injected Airflow failures above",
        "two deliberately injected Airflow failures above",
    )
    text = text.replace(
        "pending real Airflow, Kubernetes and LLM demonstrations",
        "the separate real Airflow, Kubernetes and LLM demonstration exports",
    )
    marker = "\n## Verified live execution record\n"
    text = text.split(marker)[0]
    text += marker + "\n" + "\n".join(f"- {key}: {value}" for key, value in result.items()) + "\n"
    text += "\nAll five recovery tasks succeeded. The validation failure log identified the injected missing-feature error; the quality failure log identified the explicit gate rejection. Both were followed by a successful recovery run. Historical data bytes were retrieved and verified, and the deployment/inference observations are exported in evidence/.\n"
    report.write_text(text)
    print("Report updated from inspected live evidence.")


if __name__ == "__main__":
    main()
