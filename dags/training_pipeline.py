import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

from airflow.sdk import dag, get_current_context, task

from training import train as stages

ROOT = Path(os.environ.get("MLOPS_REPO_ROOT", "/opt/project"))
TRACKING = os.environ.get("MLOPS_MLFLOW_TRACKING_URI", "http://mlflow:5000")


def workdir():
    context = get_current_context()
    safe_id = hashlib.sha256(context["run_id"].encode()).hexdigest()[:24]
    return ROOT / "artifacts" / safe_id


@dag(
    dag_id="iris_training",
    schedule=None,
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    max_active_runs=1,
    is_paused_upon_creation=False,
    default_args={"retries": 1, "retry_delay": timedelta(seconds=10)},
    tags=["mlops", "iris", "assignment-3"],
)
def training_pipeline():
    @task
    def validate():
        conf = get_current_context()["dag_run"].conf or {}
        stages.validate(ROOT, workdir(), conf.get("inject_failure", "none"))

    @task
    def preprocess():
        stages.preprocess(workdir())

    @task
    def train():
        stages.train(workdir())

    @task
    def evaluate_mlflow():
        stages.evaluate(workdir(), TRACKING, get_current_context()["run_id"])

    @task
    def quality_gate():
        conf = get_current_context()["dag_run"].conf or {}
        stages.quality_gate(
            workdir(),
            TRACKING,
            float(conf.get("quality_threshold", 0.90)),
            conf.get("inject_failure", "none"),
        )

    validate() >> preprocess() >> train() >> evaluate_mlflow() >> quality_gate()


training_pipeline()
