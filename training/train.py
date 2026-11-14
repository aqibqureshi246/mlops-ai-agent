import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

LOG = logging.getLogger(__name__)
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def dump(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(root: Path, work: Path, inject_failure: str = "none") -> dict:
    path = root / "data/iris.csv"
    if inject_failure == "validation":
        raise ValueError("Injected validation failure: missing required feature petal_width")
    frame = pd.read_csv(path)
    if list(frame.columns) != FEATURES + ["target"]:
        raise ValueError("Dataset schema mismatch")
    if len(frame) < 100 or frame.isna().any().any():
        raise ValueError("Dataset requires >=100 complete rows")
    values = frame[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Features must be finite and positive")
    if set(frame.target.unique()) != {0, 1, 2} or frame.target.value_counts().min() < 10:
        raise ValueError("Expected three sufficiently represented classes")
    pointer = yaml.safe_load((root / "data/iris.csv.dvc").read_text())["outs"][0]
    actual = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    if pointer.get("md5") != actual:
        raise ValueError(
            "Dataset hash disagrees with DVC pointer; run dvc checkout or version the change"
        )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(
        [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
            "--",
            "agent",
            "api",
            "training",
            "dags",
            "scripts",
            "data",
            "requirements",
            "pyproject.toml",
            "uv.lock",
            "dvc.yaml",
            "docker",
            "Dockerfile",
            "compose.yaml",
            "kubernetes",
            ".github",
            ".dvc/config",
            ".dvcignore",
        ],
        cwd=root,
        text=True,
    ).strip()
    if dirty:
        raise ValueError(
            "Tracked source or dataset metadata is modified; commit it before training"
        )
    provenance = {
        "data.git_commit": commit,
        "data.hash": actual,
        "data.pointer": "data/iris.csv.dvc",
        "code.git_commit": commit,
    }
    work.mkdir(parents=True, exist_ok=True)
    frame.to_csv(work / "validated.csv", index=False, lineterminator="\n")
    dump(work / "provenance.json", provenance)
    LOG.info("Validated %s rows; dataset_hash=%s commit=%s", len(frame), actual, commit)
    return provenance


def preprocess(work: Path, seed: int = 42):
    frame = pd.read_csv(work / "validated.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame.target, test_size=0.2, random_state=seed, stratify=frame.target
    )
    joblib.dump((x_train, x_test, y_train, y_test), work / "split.joblib")
    dump(
        work / "split.json",
        {"seed": seed, "test_size": 0.2, "train_rows": len(y_train), "test_rows": len(y_test)},
    )
    LOG.info("Split complete; scaler will fit training rows only")


def train(work: Path, c: float = 1.0, seed: int = 42):
    x_train, _, y_train, _ = joblib.load(work / "split.joblib")
    model = make_pipeline(
        StandardScaler(), LogisticRegression(C=c, random_state=seed, max_iter=500)
    )
    model.fit(x_train, y_train)
    joblib.dump(model, work / "model.joblib")
    dump(
        work / "params.json",
        {
            "C": c,
            "seed": seed,
            "max_iter": 500,
            "model": "StandardScaler+LogisticRegression",
            "test_size": 0.2,
        },
    )
    LOG.info("Training complete")


def evaluate(work: Path, tracking_uri: str, airflow_run_id: str = "local") -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("iris-classification")
    model = joblib.load(work / "model.joblib")
    _, x_test, _, y_test = joblib.load(work / "split.joblib")
    predicted = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predicted)),
        "f1_macro": float(f1_score(y_test, predicted, average="macro")),
    }
    with mlflow.start_run(run_name=f"iris-{airflow_run_id}") as run:
        mlflow.log_params(load(work / "params.json"))
        mlflow.log_metrics(metrics)
        mlflow.set_tags(
            {
                **load(work / "provenance.json"),
                "quality.passed": "pending",
                "airflow.run_id": airflow_run_id,
            }
        )
        mlflow.log_artifact(str(work / "provenance.json"))
        mlflow.log_artifact(str(work / "split.json"))
        mlflow.log_artifact(str(work / "model.joblib"), artifact_path="model_bundle")
        result = {"run_id": run.info.run_id, "metrics": metrics}
    dump(work / "evaluation.json", result)
    LOG.info("Evaluation run=%s metrics=%s", result["run_id"], metrics)
    return result


def quality_gate(
    work: Path, tracking_uri: str, threshold: float = 0.90, inject_failure: str = "none"
) -> dict:
    if not 0 <= threshold <= 1:
        raise ValueError("Quality threshold must be in [0,1]")
    approved = (work / "approved").resolve()
    if not approved.is_relative_to(work.resolve()):
        raise ValueError("Approved bundle escapes the run directory")
    if approved.exists():
        shutil.rmtree(approved)
    result = load(work / "evaluation.json")
    passed = result["metrics"]["f1_macro"] >= threshold and inject_failure != "quality"
    client = mlflow.MlflowClient(tracking_uri=tracking_uri)
    client.set_tag(result["run_id"], "quality.passed", str(passed).lower())
    client.set_tag(result["run_id"], "quality.threshold", str(threshold))
    dump(work / "quality.json", {"passed": passed, "threshold": threshold, **result})
    client.log_artifact(result["run_id"], str(work / "quality.json"))
    if not passed:
        raise ValueError(
            f"Quality gate rejected candidate: F1={result['metrics']['f1_macro']:.4f}, threshold={threshold}; injected={inject_failure == 'quality'}"
        )
    approved = work / "approved"
    approved.mkdir(exist_ok=True)
    shutil.copy2(work / "model.joblib", approved / "model.joblib")
    metadata = {
        **result,
        "quality_passed": True,
        "features": FEATURES,
        "classes": ["setosa", "versicolor", "virginica"],
        "provenance": load(work / "provenance.json"),
        "model_sha256": hashlib.sha256((approved / "model.joblib").read_bytes()).hexdigest(),
    }
    dump(approved / "metadata.json", metadata)
    LOG.info("Quality gate passed; approved bundle=%s", approved)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage", choices=["validate", "preprocess", "train", "evaluate", "quality", "all"]
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--work", type=Path, default=Path("artifacts/local"))
    parser.add_argument("--tracking-uri", default="http://localhost:5000")
    parser.add_argument("--c", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument(
        "--inject-failure", choices=["none", "validation", "quality"], default="none"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    stage, work = args.stage, args.work
    if stage in ("validate", "all"):
        validate(args.root, work, args.inject_failure)
    if stage in ("preprocess", "all"):
        preprocess(work)
    if stage in ("train", "all"):
        train(work, args.c)
    if stage in ("evaluate", "all"):
        evaluate(work, args.tracking_uri)
    if stage in ("quality", "all"):
        quality_gate(work, args.tracking_uri, args.threshold, args.inject_failure)


if __name__ == "__main__":
    main()
