"""Append a verification record only after inspecting complete live evidence."""

import json
from datetime import UTC, datetime
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


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
    tasks = next(
        e["data"]["tasks"] for e in success["evidence"] if e["tool"] == "airflow_task_instances"
    )
    if len(tasks) != 5 or any(t["state"] != "success" for t in tasks):
        raise SystemExit("Recovery training did not finish successfully")
    result = {
        "verified_at": datetime.now(UTC).isoformat(),
        "recovery_run": success["run_id"],
        "validation_failure_run": validation["run_id"],
        "quality_failure_run": quality["run_id"],
        "dataset_hash": reproduced["retrieved_hash"],
        "complete_conversations": len(conversations),
    }
    Path("evidence/verification-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    report = Path("docs/technical-report.md")
    text = report.read_text()
    header = text.index("## Validation record")
    table = text.index("| Evidence |", header)
    text = (
        text[:header]
        + "## Validation record\n\nThe manual live workflow completed the observations below. Actual run IDs and verification time are recorded at the end of this report.\n\n"
        + text[table:]
    )
    text = text.replace("Pending live execution", "Verified; see exported evidence")
    text = text.replace("Pending execution", "Passed; see automated-tests.json")
    text = text.replace(
        "These are planned reproducible investigations until the corresponding real exports exist.",
        "Both failure investigations were executed and their root causes were verified in the exported logs.",
    )
    marker = "\n## Verified live execution record\n"
    text = text.split(marker)[0]
    text += marker + "\n" + "\n".join(f"- {key}: {value}" for key, value in result.items()) + "\n"
    text += "\nAll five recovery tasks succeeded. The validation failure log identified the injected missing-feature error; the quality failure log identified the explicit gate rejection. Both were followed by a successful recovery run. Historical data bytes were retrieved and verified, and the deployment/inference observations are exported in evidence/.\n"
    report.write_text(text)
    print("Report updated from inspected live evidence.")


if __name__ == "__main__":
    main()
