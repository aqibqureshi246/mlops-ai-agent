import json
import time
from unittest.mock import Mock

import httpx
import pytest
from conftest import ScriptedPlanner

from agent.config import Settings
from agent.orchestrator import Orchestrator, assess, planner_observation
from agent.registry import Registry, Tool
from agent.schemas import EmptyInput, Evidence


def registry_with(handler, mutating=False):
    registry = Registry()
    registry.add(Tool("inspect", "test tool", EmptyInput, handler, mutating))
    return registry


def test_structured_tool_selection_executes_and_returns_real_evidence():
    handler = Mock(return_value={"replicas": 2})
    registry = registry_with(handler)
    response = Orchestrator(registry, ScriptedPlanner([("inspect", {})]), Settings()).chat(
        "Check health"
    )
    assert response.evidence[0].data == {"replicas": 2}
    assert '"replicas": 2' in response.answer
    handler.assert_called_once_with()


def test_unavailable_service_never_becomes_healthy_result():
    registry = registry_with(Mock(side_effect=ConnectionError("sensitive-url")))
    result = Orchestrator(registry, ScriptedPlanner([("inspect", {})]), Settings()).chat(
        "Check health"
    )
    assert result.evidence[0].status == "error"
    assert result.evidence[0].data == {}
    assert "sensitive-url" not in result.model_dump_json()
    assert result.uncertainties


def test_timeout_is_unknown_not_success():
    registry = registry_with(Mock(side_effect=httpx.ReadTimeout("secret")))
    result = registry.execute("inspect", {}, "r1")
    assert result.status == "timeout" and "unknown" in result.error


def test_invalid_input_rejected_before_execution():
    handler = Mock()
    result = registry_with(handler).execute("inspect", {"shell": "rm -rf /"}, "r1")
    assert result.status == "error"
    handler.assert_not_called()


def test_unknown_tool_rejected():
    assert Registry().execute("delete_namespace", {}, "r").status == "error"


def test_unauthorized_modification_never_reaches_adapter():
    handler = Mock()
    registry = registry_with(handler, mutating=True)
    assert registry.execute("inspect", {}, "r").status == "denied"
    handler.assert_not_called()


def test_confirmation_is_bound_to_owner_and_single_use():
    handler = Mock(return_value={"requested": True})
    registry = registry_with(handler, mutating=True)
    action = registry.propose("inspect", {}, "alice")
    with pytest.raises(PermissionError):
        registry.confirm(action["confirmation_token"], "bob", True, "r")
    assert registry.confirm(action["confirmation_token"], "alice", True, "r").status == "ok"
    with pytest.raises(PermissionError):
        registry.confirm(action["confirmation_token"], "alice", True, "r")
    handler.assert_called_once()


def test_expired_and_declined_confirmations_do_not_execute():
    handler = Mock()
    registry = registry_with(handler, mutating=True)
    action = registry.propose("inspect", {}, "alice")
    list(registry.pending.values())[0]["expires"] = time.monotonic() - 1
    with pytest.raises(PermissionError):
        registry.confirm(action["confirmation_token"], "alice", True, "r")
    action = registry.propose("inspect", {}, "alice")
    assert registry.confirm(action["confirmation_token"], "alice", False, "r").status == "denied"
    handler.assert_not_called()


def test_llm_cannot_confirm_mutation_through_tool_arguments():
    handler = Mock()
    registry = registry_with(handler, mutating=True)
    response = Orchestrator(
        registry, ScriptedPlanner([("inspect", {"confirmed": True})]), Settings()
    ).chat("Restart")
    assert response.evidence[0].status == "error"
    handler.assert_not_called()


def test_mutation_produces_reviewable_action_without_execution():
    handler = Mock()
    registry = registry_with(handler, mutating=True)
    response = Orchestrator(registry, ScriptedPlanner([("inspect", {})]), Settings()).chat(
        "Restart"
    )
    assert response.evidence[0].status == "denied"
    assert response.pending_actions[0]["tool"] == "inspect"
    handler.assert_not_called()


def test_model_outage_returns_no_invented_evidence():
    planner = Mock()
    planner.next_calls.side_effect = RuntimeError("API key")
    result = Orchestrator(Registry(), planner, Settings()).chat("Best model?")
    assert result.evidence == [] and "unknown" in result.answer
    assert any("planner unavailable" in x for x in result.uncertainties)


def test_audit_log_has_required_fields_without_payload(caplog):
    registry = registry_with(lambda: {"password": "never-log-this", "count": 2})
    with caplog.at_level("INFO", logger="mlops.audit"):
        result = registry.execute("inspect", {}, "request-42")
    audit = json.loads(caplog.records[-1].message)
    assert {"tool", "status", "duration_ms", "request_id"} <= audit.keys()
    assert "never-log-this" not in caplog.text
    assert result.data["password"] == "[REDACTED]"


def test_duplicate_calls_are_bounded():
    handler = Mock(return_value={})
    planner = ScriptedPlanner([("inspect", {})], [("inspect", {})])
    result = Orchestrator(registry_with(handler), planner, Settings()).chat("Check")
    assert result.evidence[-1].status == "error"
    handler.assert_called_once()


def test_missing_observation_prompts_the_planner_again_without_scripted_calls():
    handler = Mock(return_value={"replicas": 2})
    planner = ScriptedPlanner([], [("inspect", {})])
    result = Orchestrator(registry_with(handler), planner, Settings()).chat(
        "Check health", required_tools={"inspect"}
    )
    assert result.evidence[0].data == {"replicas": 2}
    handler.assert_called_once_with()
    assert not result.uncertainties


def test_required_observations_never_fall_back_to_automatic_tool_execution():
    handler = Mock(return_value={"replicas": 2})
    result = Orchestrator(registry_with(handler), ScriptedPlanner(), Settings(max_rounds=2)).chat(
        "Check health", required_tools={"inspect"}
    )
    handler.assert_not_called()
    assert not result.evidence
    assert any(
        "Required observations remain unavailable: inspect" in u for u in result.uncertainties
    )


def test_explicit_observation_contract_limits_the_models_available_tools():
    handler = Mock(return_value={"replicas": 2})
    registry = registry_with(handler)
    registry.add(Tool("unrelated", "Unneeded tool", EmptyInput, Mock()))
    planner = Mock(wraps=ScriptedPlanner([("inspect", {})]))
    result = Orchestrator(registry, planner, Settings()).chat(
        "Check health", required_tools={"inspect"}
    )
    definitions = planner.next_calls.call_args.args[1]
    assert {d["function"]["name"] for d in definitions} == {"inspect"}
    assert result.evidence[0].data == {"replicas": 2}
    handler.assert_called_once_with()


def test_observation_requirements_cannot_authorize_mutations():
    handler = Mock()
    with pytest.raises(ValueError, match="read-only"):
        Orchestrator(registry_with(handler, mutating=True), ScriptedPlanner(), Settings()).chat(
            "Check health", required_tools={"inspect"}
        )
    handler.assert_not_called()


def test_planner_comparison_is_compact_but_preserves_full_audit_evidence():
    best = {
        "run_id": "a" * 32,
        "status": "FINISHED",
        "metrics": {"f1_macro": 0.95},
        "params": {"C": "3.0"},
        "provenance": {"data.git_commit": "b" * 40, "data.pointer": "data/iris.csv.dvc"},
    }
    item = Evidence(
        id="e1",
        tool="mlflow_best_run",
        status="ok",
        request_id="r",
        data={"comparison": [best], "best": best},
    )
    original = item.model_dump_json()
    planned = json.loads(planner_observation(item))["data"]
    assert planned["best"] == best
    assert planned["comparison"][0] == {
        "run_id": "a" * 32,
        "status": "FINISHED",
        "metrics": {"f1_macro": 0.95},
    }
    assert item.model_dump_json() == original


def test_integration_correlates_all_four_services_without_retraining():
    registry = Registry()
    data = {
        "mlflow_best_run": {
            "metric": "f1_macro",
            "recent_window": 5,
            "caveat": "Candidate ranking only",
            "best": {
                "metrics": {"f1_macro": 0.95},
                "run_id": "a" * 32,
                "start_time_ms": time.time() * 1000,
                "provenance": {
                    "quality.passed": "true",
                    "data.git_commit": "abc",
                    "data.hash": "hash",
                },
            },
        },
        "dvc_reproduction_steps": {"git_commit": "abc", "recorded_hash": "hash"},
        "kubernetes_deployments": {
            "deployments": [
                {
                    "name": "prediction-api",
                    "desired": 2,
                    "ready": 2,
                    "generation": 1,
                    "observed_generation": 1,
                    "model_run_id": "a" * 32,
                    "images": ["prediction:test"],
                }
            ]
        },
        "kubernetes_pods": {
            "pods": [
                {
                    "name": "prediction-api-test",
                    "phase": "Running",
                    "conditions": [{"type": "Ready", "status": "True"}],
                    "containers": [{"restarts": 0}],
                }
            ]
        },
        "airflow_dag_runs": {"runs": [{"state": "success"}]},
    }
    for name, value in data.items():
        registry.add(Tool(name, name, EmptyInput, lambda value=value: value))
    response = Orchestrator(registry, ScriptedPlanner([(n, {}) for n in data]), Settings()).chat(
        "Find best model, data, deployment and retraining"
    )
    assert len(response.evidence) == 5
    assert not response.pending_actions
    assert not response.uncertainties
    assert "No retraining is automatically initiated" in response.recommendations[-1]


def test_freshness_and_hash_mismatch_are_evidence_based():
    evidence = [
        Evidence(
            id="e1",
            tool="mlflow_best_run",
            status="ok",
            request_id="r",
            data={
                "best": {
                    "run_id": "a" * 32,
                    "start_time_ms": 1000,
                    "provenance": {
                        "quality.passed": "false",
                        "data.git_commit": "abc",
                        "data.hash": "old",
                    },
                }
            },
        ),
        Evidence(
            id="e2",
            tool="dvc_reproduction_steps",
            status="ok",
            request_id="r",
            data={"git_commit": "abc", "recorded_hash": "different"},
        ),
    ]
    _, recommendations = assess(evidence, 30)
    assert any("freshness" in r for r in recommendations)
    assert any("disagrees" in r for r in recommendations)
    assert any("quality gate" in r for r in recommendations)


def test_malformed_service_output_is_not_reported_as_evidence():
    result = registry_with(lambda: ["unexpected"]).execute("inspect", {}, "r")
    assert result.status == "error" and result.data == {}
