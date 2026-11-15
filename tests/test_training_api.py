import json
from unittest.mock import Mock

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.datasets import load_iris

from api.main import create_app
from training import train as stages


@pytest.fixture
def trained(tmp_path):
    frame = load_iris(as_frame=True).frame
    frame.columns = stages.FEATURES + ["target"]
    frame.to_csv(tmp_path / "validated.csv", index=False)
    stages.dump(tmp_path / "provenance.json", {"data.hash": "test-hash", "data.git_commit": "abc"})
    stages.preprocess(tmp_path)
    stages.train(tmp_path)
    return tmp_path


def test_training_split_is_seeded_and_does_not_overlap(trained):
    first = joblib.load(trained / "split.joblib")
    stages.preprocess(trained)
    second = joblib.load(trained / "split.joblib")
    assert first[0].equals(second[0])
    assert set(first[0].index).isdisjoint(set(first[1].index))
    assert len(first[0]) == 120 and len(first[1]) == 30


def test_validation_injection_stops_before_writing_outputs(tmp_path):
    with pytest.raises(ValueError, match="Injected validation failure"):
        stages.validate(tmp_path, tmp_path / "work", "validation")
    assert not (tmp_path / "work").exists()


def test_schema_validation_rejects_missing_column(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/iris.csv").write_text("wrong,target\n1,0\n")
    with pytest.raises(ValueError, match="schema mismatch"):
        stages.validate(tmp_path, tmp_path / "work")


def test_quality_gate_failure_blocks_approved_bundle(trained, monkeypatch):
    client = Mock()
    monkeypatch.setattr(stages.mlflow, "MlflowClient", lambda **kwargs: client)
    stages.dump(trained / "evaluation.json", {"run_id": "a" * 32, "metrics": {"f1_macro": 0.4}})
    with pytest.raises(ValueError, match="rejected"):
        stages.quality_gate(trained, "http://mock", 0.9)
    assert not (trained / "approved").exists()
    client.set_tag.assert_any_call("a" * 32, "quality.passed", "false")


def test_quality_gate_resolves_remote_artifacts_after_task_process_reset(trained, monkeypatch):
    from mlflow.store.artifact.mlflow_artifacts_repo import MlflowArtifactsRepository

    original = stages.mlflow.get_tracking_uri()
    resolved = []
    client = Mock()

    def upload(run_id, path):
        repository = MlflowArtifactsRepository(f"mlflow-artifacts:/1/{run_id}/artifacts")
        resolved.append(repository.artifact_uri)

    client.log_artifact.side_effect = upload
    monkeypatch.setattr(stages.mlflow, "MlflowClient", lambda **kwargs: client)
    stages.dump(trained / "evaluation.json", {"run_id": "a" * 32, "metrics": {"f1_macro": 0.95}})
    try:
        stages.mlflow.set_tracking_uri((trained / "unrelated-local-tracking").as_uri())
        stages.quality_gate(trained, "http://mlflow:5000")
        assert resolved == [
            "http://mlflow:5000/api/2.0/mlflow-artifacts/artifacts/1/" + "a" * 32 + "/artifacts"
        ]
        assert (trained / "approved/metadata.json").exists()
    finally:
        stages.mlflow.set_tracking_uri(original)


def test_approved_model_predicts_and_reports_provenance(trained, monkeypatch):
    monkeypatch.setattr(stages.mlflow, "MlflowClient", lambda **kwargs: Mock())
    stages.dump(trained / "evaluation.json", {"run_id": "a" * 32, "metrics": {"f1_macro": 0.95}})
    stages.quality_gate(trained, "http://mock")
    with TestClient(create_app(trained / "approved")) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 200
        result = client.post(
            "/predict",
            json={"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2},
        )
        assert result.status_code == 200
        assert result.json()["class_name"] == "setosa"
        assert client.get("/version").json()["dataset"]["data.hash"] == "test-hash"
        assert client.post("/predict", json={"sepal_length": -1}).status_code == 422


def test_missing_model_has_liveness_but_fails_readiness(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_corrupted_bundle_never_loads_pickle(tmp_path, monkeypatch):
    (tmp_path / "model.joblib").write_bytes(b"untrusted")
    (tmp_path / "metadata.json").write_text(
        json.dumps({"quality_passed": True, "model_sha256": "wrong"})
    )
    loader = Mock()
    monkeypatch.setattr("api.main.joblib.load", loader)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health/ready").status_code == 503
    loader.assert_not_called()


def test_real_local_mlflow_records_metrics_artifacts_and_selection(trained, tmp_path):
    import mlflow

    from agent.tools.mlflow_tools import MlflowTools

    original = mlflow.get_tracking_uri()
    uri = (tmp_path / "tracking").as_uri()
    try:
        first = stages.evaluate(trained, uri, "local-integration-test")
        stages.quality_gate(trained, uri)
        client = mlflow.MlflowClient(tracking_uri=uri)
        tools = MlflowTools(client, "iris-classification")
        details = tools.run_details(first["run_id"])
        assert details["status"] == "FINISHED"
        assert details["metrics"]["f1_macro"] >= 0.9
        assert details["provenance"]["quality.passed"] == "true"
        assert any(a["path"] == "model_bundle" for a in details["artifacts"])
        assert tools.best_run()["best"]["run_id"] == first["run_id"]
    finally:
        mlflow.set_tracking_uri(original)


def test_validation_allows_new_evidence_but_rejects_modified_training_source(tmp_path):
    import hashlib
    import subprocess

    import yaml

    (tmp_path / "data").mkdir()
    (tmp_path / "training").mkdir()
    (tmp_path / "evidence").mkdir()
    frame = load_iris(as_frame=True).frame
    frame.columns = stages.FEATURES + ["target"]
    path = tmp_path / "data/iris.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    (tmp_path / "data/iris.csv.dvc").write_text(
        yaml.safe_dump({"outs": [{"path": "iris.csv", "md5": digest}]})
    )
    source = tmp_path / "training/train.py"
    source.write_text("# committed training source\n")
    observation = tmp_path / "evidence/result.json"
    observation.write_text("{}")

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "source")
    observation.write_text('{"observed": true}')
    provenance = stages.validate(tmp_path, tmp_path / "work")
    assert provenance["code.git_commit"] == git("rev-parse", "HEAD")
    source.write_text("# modified source\n")
    with pytest.raises(ValueError, match="Tracked source"):
        stages.validate(tmp_path, tmp_path / "work-2")
