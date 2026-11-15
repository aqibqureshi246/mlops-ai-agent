import hashlib
import json
import subprocess
from types import SimpleNamespace as NS
from unittest.mock import Mock

import httpx
import pytest
import yaml
from conftest import make_run

from agent.tools.airflow_tools import AirflowTools
from agent.tools.dvc_tools import DvcTools
from agent.tools.kubernetes_tools import KubernetesTools
from agent.tools.mlflow_tools import MlflowTools


def test_dvc_version_matches_real_file_and_git_commit(tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "data").mkdir()
    data = tmp_path / "data/iris.csv"
    data.write_bytes(b"x,y\n1,2\n")
    digest = hashlib.md5(data.read_bytes(), usedforsecurity=False).hexdigest()
    pointer = tmp_path / "data/iris.csv.dvc"
    pointer.write_text(yaml.safe_dump({"outs": [{"path": "iris.csv", "md5": digest}]}))
    subprocess.run(["git", "-C", str(tmp_path), "add", "."], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.invalid",
            "commit",
            "-qm",
            "data",
        ],
        check=True,
    )
    tools = DvcTools(tmp_path)
    result = tools.dataset_version()
    assert result["matches_pointer"] and len(result["git_commit"]) == 40
    historical = tools.reproduction_steps(result["git_commit"])
    assert historical["recorded_hash"] == digest
    data.write_bytes(b"changed")
    assert not tools.dataset_version()["matches_pointer"]


def test_dvc_rejects_traversal_and_revision_injection(tmp_path):
    tools = DvcTools(tmp_path)
    with pytest.raises(ValueError):
        tools.dataset_version("../outside.dvc")
    with pytest.raises(ValueError):
        tools.compare_versions("HEAD; rm -rf /")
    with pytest.raises(ValueError):
        tools.reproduction_steps("--help")


def test_dvc_status_and_comparison_parse_structured_output(tmp_path, monkeypatch):
    tools = DvcTools(tmp_path)
    monkeypatch.setattr(tools, "_run", lambda *args: '{"data/iris.csv": "modified"}')
    assert tools.data_status()["changes"]["data/iris.csv"] == "modified"
    monkeypatch.setattr(tools, "_revision", lambda rev: rev)
    assert tools.compare_versions("a", "b")["new_commit"] == "b"


def test_mlflow_best_run_compares_recent_finished_candidates():
    client = Mock()
    client.get_experiment_by_name.return_value = NS(experiment_id="1")
    client.search_runs.return_value = [
        make_run("a" * 32, 0.9),
        make_run("b" * 32, 0.95),
        make_run("c" * 32, 1.0, "FAILED"),
    ]
    result = MlflowTools(client, "iris").best_run()
    assert result["best"]["run_id"] == "b" * 32
    assert len(result["comparison"]) == 3
    assert "production readiness" in result["caveat"]


def test_mlflow_empty_runs_does_not_invent_candidate():
    client = Mock()
    client.get_experiment_by_name.return_value = None
    assert MlflowTools(client, "missing").best_run()["best"] is None


def test_mlflow_details_include_provenance_and_artifacts():
    client = Mock()
    client.get_run.return_value = make_run(tags={"data.hash": "123", "password": "secret"})
    client.list_artifacts.return_value = [NS(path="model_bundle", is_dir=True)]
    result = MlflowTools(client, "iris").run_details("a" * 32)
    assert result["provenance"] == {"data.hash": "123"}
    assert result["artifacts"][0]["path"] == "model_bundle"


def test_registry_mlflow_details_resolve_remote_artifacts_from_local_default(tmp_path, monkeypatch):
    from uuid import uuid4

    import mlflow
    from mlflow.store.artifact.http_artifact_repo import HttpArtifactRepository

    from agent.config import Settings
    from agent.factory import build_registry

    original = mlflow.get_tracking_uri()
    uri = "http://tracking.example.invalid:5000"
    client = mlflow.MlflowClient(tracking_uri=uri)
    run_id = uuid4().hex
    run = make_run(run_id)
    run.info.artifact_uri = f"mlflow-artifacts:/1/{run_id}/artifacts"
    monkeypatch.setattr(client._tracking_client, "get_run", lambda run_id: run)
    monkeypatch.setattr(mlflow, "MlflowClient", lambda **kwargs: client)
    monkeypatch.setattr(
        HttpArtifactRepository,
        "list_artifacts",
        lambda self, path=None: [NS(path="quality.json", is_dir=False)],
    )
    try:
        mlflow.set_tracking_uri((tmp_path / "unrelated-local-tracking").as_uri())
        registry = build_registry(Settings(mlflow_tracking_uri=uri))
        result = registry.execute("mlflow_run_details", {"run_id": run_id}, "artifact-test")
        assert result.status == "ok", result.error
        assert result.data["artifacts"] == [{"path": "quality.json", "is_dir": False}]
    finally:
        mlflow.set_tracking_uri(original)


def test_airflow_uses_authenticated_v2_api_and_returns_real_failed_tasks():
    calls = []

    def handler(request):
        calls.append(request)
        if request.url.path == "/auth/token":
            return httpx.Response(200, json={"access_token": "jwt"})
        assert request.headers["Authorization"] == "Bearer jwt"
        assert request.url.path.startswith("/api/v2/dags/iris_training/")
        return httpx.Response(
            200,
            json={
                "task_instances": [
                    {"task_id": "validate", "state": "failed", "try_number": 2},
                    {"task_id": "train", "state": "upstream_failed"},
                ]
            },
        )

    client = httpx.Client(base_url="http://airflow", transport=httpx.MockTransport(handler))
    tools = AirflowTools("http://airflow", "student", "secret", "iris_training", client=client)
    result = tools.task_instances("iris_training", "manual__one")
    assert len(result["failed_tasks"]) == 2
    assert len(calls) == 2


def test_airflow_unapproved_dag_rejected_without_network():
    client = Mock()
    tools = AirflowTools("http://airflow", "student", "secret", "iris_training", client=client)
    with pytest.raises(PermissionError):
        tools.trigger_training("admin_cleanup")
    client.post.assert_not_called()


def test_airflow_trigger_is_acknowledgement_not_claimed_success():
    def handler(request):
        if request.url.path == "/auth/token":
            return httpx.Response(200, json={"access_token": "jwt"})
        payload = json.loads(request.content)
        assert payload["conf"]["quality_threshold"] == 0.9
        assert payload["logical_date"]
        return httpx.Response(200, json={"dag_run_id": payload["dag_run_id"], "state": "queued"})

    client = httpx.Client(base_url="http://airflow", transport=httpx.MockTransport(handler))
    result = AirflowTools(
        "http://airflow", "s", "p", "iris_training", client=client
    ).trigger_training()
    assert result["state"] == "queued"
    assert "terminal state" in result["note"]


def test_kubernetes_namespace_rejection_precedes_api_call():
    core, apps = Mock(), Mock()
    tools = KubernetesTools(core, apps, "mlops", "prediction-api")
    with pytest.raises(PermissionError):
        tools.restart_deployment(namespace="kube-system")
    with pytest.raises(PermissionError):
        tools.pods(namespace="default")
    apps.patch_namespaced_deployment.assert_not_called()
    core.list_namespaced_pod.assert_not_called()


def test_kubernetes_restart_cannot_target_other_deployment():
    tools = KubernetesTools(Mock(), Mock(), "mlops", "prediction-api")
    with pytest.raises(PermissionError):
        tools.restart_deployment(deployment="postgres")
    tools.apps.patch_namespaced_deployment.assert_not_called()


def test_kubernetes_pods_expose_readiness_restarts_and_waiting_reason():
    core = Mock()
    core.list_namespaced_pod.return_value.items = [
        NS(
            metadata=NS(name="prediction-api-x"),
            status=NS(
                phase="Running",
                conditions=[NS(type="Ready", status="False", reason="ContainersNotReady")],
                container_statuses=[
                    NS(
                        name="prediction-api",
                        ready=False,
                        restart_count=3,
                        state=NS(waiting=NS(reason="CrashLoopBackOff")),
                        last_state=None,
                    )
                ],
            ),
        )
    ]
    result = KubernetesTools(core, Mock(), "mlops", "prediction-api").pods()
    assert result["pods"][0]["containers"][0]["restarts"] == 3
    assert result["pods"][0]["conditions"][0]["status"] == "False"


def test_logs_of_other_workloads_rejected():
    core = Mock()
    core.read_namespaced_pod.return_value.metadata.labels = {"app": "database"}
    with pytest.raises(PermissionError):
        KubernetesTools(core, Mock(), "mlops", "prediction-api").pod_logs("database-1")
    core.read_namespaced_pod_log.assert_not_called()


def test_compare_experiments_reports_each_recorded_window():
    client = Mock()
    client.get_experiment.side_effect = [NS(name="first"), NS(name="second")]
    client.search_runs.side_effect = [[make_run("a" * 32, 0.9)], [make_run("b" * 32, 0.95)]]
    result = MlflowTools(client, "iris").compare_experiments(["1", "2"])
    assert [e["name"] for e in result["experiments"]] == ["first", "second"]
    assert result["experiments"][1]["recent_runs"][0]["metrics"]["f1_macro"] == 0.95


def test_factory_never_falls_back_to_admin_credentials_for_restart(tmp_path, monkeypatch):
    from agent.config import Settings
    from agent.factory import build_registry
    from kubernetes import client, config
    from kubernetes.config.config_exception import ConfigException

    def not_in_cluster(**kwargs):
        raise ConfigException("Not in cluster")

    loader = Mock()
    core = Mock()
    core.list_namespaced_pod.return_value.items = []
    monkeypatch.setattr(config, "load_incluster_config", not_in_cluster)
    monkeypatch.setattr(config, "load_kube_config", loader)
    monkeypatch.setattr(client, "CoreV1Api", lambda api_client: core)
    monkeypatch.setattr(client, "AppsV1Api", lambda api_client: Mock())
    settings = Settings(repo_root=tmp_path, kubeconfig=tmp_path / "reader", kube_write_config=None)
    registry = build_registry(settings)
    assert registry.execute("kubernetes_pods", {}, "r").status == "ok"
    assert loader.call_args.kwargs["context"] == "mlops-agent-reader"
    assert (
        registry.execute("kubernetes_restart_deployment", {}, "r", authorized=True).status
        == "denied"
    )
    assert loader.call_count == 1
