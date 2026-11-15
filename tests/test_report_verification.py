import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "update_report", Path(__file__).resolve().parents[1] / "scripts/update_report.py"
)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)
verify_pipeline_states = report.verify_pipeline_states

TASKS = ("validate", "preprocess", "train", "evaluate_mlflow", "quality_gate")


def observation(states, status="ok"):
    return {
        "evidence": [
            {
                "tool": "airflow_task_instances",
                "status": status,
                "data": {
                    "tasks": [{"task_id": task, "state": state} for task, state in states.items()]
                },
            }
        ]
    }


@pytest.fixture
def executions():
    recovery = dict.fromkeys(TASKS, "success")
    validation = dict.fromkeys(TASKS, "upstream_failed")
    validation["validate"] = "failed"
    quality = {**recovery, "quality_gate": "failed"}
    return [observation(states) for states in (recovery, validation, quality)]


def test_report_accepts_verified_failure_boundaries_and_recovery(executions):
    verify_pipeline_states(*executions)


@pytest.mark.parametrize("index", [0, 1, 2])
def test_report_rejects_partial_or_wrong_task_outcomes(executions, index):
    # A matching error phrase alone cannot prove the expected execution path.
    executions[index]["evidence"][0]["data"]["tasks"].pop()
    with pytest.raises(SystemExit):
        verify_pipeline_states(*executions)


def test_report_rejects_unsuccessful_task_observation(executions):
    executions[0]["evidence"][0]["status"] = "error"
    with pytest.raises(SystemExit):
        verify_pipeline_states(*executions)
