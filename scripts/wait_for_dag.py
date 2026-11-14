import time
from uuid import uuid4

from agent.config import Settings
from agent.factory import build_registry


def main():
    registry = build_registry(Settings())
    for _ in range(60):
        result = registry.execute("airflow_list_dags", {}, uuid4().hex)
        if result.status == "ok" and any(
            d["dag_id"] == "iris_training" and not d["is_paused"] for d in result.data["dags"]
        ):
            print("Training DAG is parsed, available and unpaused.")
            return
        time.sleep(3)
    raise SystemExit("Training DAG did not become available")


if __name__ == "__main__":
    main()
