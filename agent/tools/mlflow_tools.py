import math

from pydantic import Field

from agent.schemas import StrictInput


class RunsInput(StrictInput):
    limit: int = Field(default=5, ge=1, le=50)


class RunInput(StrictInput):
    run_id: str = Field(pattern=r"^[a-fA-F0-9]{32}$")


class BestInput(RunsInput):
    metric: str = Field(default="f1_macro", pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,50}$")
    maximize: bool = True


class CompareRunsInput(StrictInput):
    run_ids: list[str] = Field(min_length=2, max_length=10)


class ExperimentCompareInput(StrictInput):
    experiment_ids: list[str] = Field(min_length=2, max_length=5)


class MlflowTools:
    def __init__(self, client, experiment_name: str):
        self.client = client
        self.experiment_name = experiment_name

    @staticmethod
    def _run(run) -> dict:
        # Emit only intentionally recorded provenance tags, not arbitrary user tags.
        return {
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time_ms": run.info.start_time,
            "end_time_ms": run.info.end_time,
            "metrics": {k: v for k, v in run.data.metrics.items() if math.isfinite(v)},
            "params": dict(run.data.params),
            "provenance": {
                k: v
                for k, v in run.data.tags.items()
                if k.startswith(("data.", "quality.", "code.", "airflow."))
            },
        }

    def list_experiments(self) -> dict:
        return {
            "experiments": [
                {"id": e.experiment_id, "name": e.name}
                for e in self.client.search_experiments(max_results=100)
            ]
        }

    def list_runs(self, limit: int = 5) -> dict:
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            return {"experiment": self.experiment_name, "runs": []}
        runs = self.client.search_runs(
            [experiment.experiment_id],
            max_results=limit,
            order_by=["attributes.start_time DESC"],
        )
        return {"experiment": self.experiment_name, "runs": [self._run(r) for r in runs]}

    def run_details(self, run_id: str) -> dict:
        run = self._run(self.client.get_run(run_id))
        run["artifacts"] = [
            {"path": a.path, "is_dir": a.is_dir} for a in self.client.list_artifacts(run_id)
        ]
        return run

    def best_run(self, metric: str = "f1_macro", maximize: bool = True, limit: int = 5) -> dict:
        recent = self.list_runs(limit)["runs"]
        eligible = [r for r in recent if r["status"] == "FINISHED" and metric in r["metrics"]]
        best = (
            (max if maximize else min)(eligible, key=lambda r: r["metrics"][metric])
            if eligible
            else None
        )
        return {
            "metric": metric,
            "maximize": maximize,
            "recent_window": limit,
            "comparison": recent,
            "best": best,
            "caveat": "Best measured candidate in this recent window; production readiness requires separate checks.",
        }

    def compare_runs(self, run_ids: list[str]) -> dict:
        validated = [RunInput(run_id=r).run_id for r in run_ids]
        return {"runs": [self.run_details(r) for r in validated]}

    def registry(self) -> dict:
        models = self.client.search_registered_models(max_results=100)
        return {
            "models": [
                {
                    "name": m.name,
                    "versions": [
                        {
                            "version": v.version,
                            "run_id": v.run_id,
                            "status": v.status,
                            "aliases": list(v.aliases or []),
                        }
                        for v in self.client.search_model_versions(
                            filter_string=f"name='{m.name}'", max_results=100
                        )
                    ],
                }
                for m in models
            ]
        }

    def compare_experiments(self, experiment_ids: list[str]) -> dict:
        comparisons = []
        for experiment_id in experiment_ids:
            experiment = self.client.get_experiment(experiment_id)
            runs = self.client.search_runs(
                [experiment_id], max_results=5, order_by=["attributes.start_time DESC"]
            )
            comparisons.append(
                {
                    "experiment_id": experiment_id,
                    "name": experiment.name,
                    "recent_runs": [self._run(run) for run in runs],
                }
            )
        return {"experiments": comparisons}
