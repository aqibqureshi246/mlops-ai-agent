"""Run inside the Airflow image to verify the real parsed DAG contract."""

from airflow.models import DagBag


def main():
    bag = DagBag(dag_folder="/opt/project/dags", include_examples=False)
    if bag.import_errors:
        raise SystemExit("DAG import errors: " + ", ".join(bag.import_errors))
    dag = bag.get_dag("iris_training")
    expected = ["validate", "preprocess", "train", "evaluate_mlflow", "quality_gate"]
    if dag is None or set(dag.task_ids) != set(expected):
        raise SystemExit("Training DAG stages do not match the required contract")
    for i, name in enumerate(expected):
        task = dag.get_task(name)
        downstream = {expected[i + 1]} if i + 1 < len(expected) else set()
        if task.downstream_task_ids != downstream or task.retries != 1:
            raise SystemExit("DAG dependency or retry contract failed")
    print("Real Airflow DAG parsed: all five stages, dependencies and retry settings verified.")


if __name__ == "__main__":
    main()
