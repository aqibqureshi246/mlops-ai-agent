import time
from uuid import uuid4

from agent.config import Settings
from agent.factory import build_registry


def main():
    registry = build_registry(Settings())
    last_observation = "No DAG observation returned"
    for _ in range(60):
        result = registry.execute("airflow_list_dags", {}, uuid4().hex)
        if result.status == "ok" and any(
            d["dag_id"] == "iris_training" and not d["is_paused"] for d in result.data["dags"]
        ):
            print("Training DAG is parsed, available and unpaused.")
            return
        observation = result.model_dump_json(exclude={"id", "request_id", "observed_at"})
        if observation != last_observation:
            print(f"Waiting for training DAG: {observation}", flush=True)
        last_observation = observation
        time.sleep(3)
    raise SystemExit(f"Training DAG did not become available. Last observation: {last_observation}")


if __name__ == "__main__":
    main()
