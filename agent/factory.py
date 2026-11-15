import os
from functools import lru_cache

from agent.config import Settings
from agent.registry import Registry, Tool
from agent.schemas import EmptyInput
from agent.tools.airflow_tools import (
    AirflowTools,
    DagInput,
    DagRunInput,
    TaskLogInput,
    TriggerInput,
)
from agent.tools.dvc_tools import CompareInput, DatasetInput, DvcTools, ReproduceInput
from agent.tools.kubernetes_tools import KubernetesTools, NamespaceInput, PodInput, RestartInput
from agent.tools.mlflow_tools import (
    BestInput,
    CompareRunsInput,
    ExperimentCompareInput,
    MlflowTools,
    RunInput,
    RunsInput,
)


def build_registry(settings: Settings) -> Registry:
    from mlflow import MlflowClient, set_tracking_uri

    # MLflow's proxied artifact resolver uses the process-level URI as well as the client.
    set_tracking_uri(settings.mlflow_tracking_uri)
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", str(int(settings.request_timeout)))
    os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "1")
    registry = Registry()
    registry.secrets = [
        settings.airflow_password.get_secret_value(),
        settings.agent_api_key.get_secret_value(),
        settings.llm_api_key.get_secret_value(),
    ]
    dvc = DvcTools(settings.repo_root, settings.request_timeout)
    mlflow = MlflowTools(
        MlflowClient(tracking_uri=settings.mlflow_tracking_uri), settings.experiment_name
    )
    airflow = AirflowTools(
        settings.airflow_url,
        settings.airflow_username,
        settings.airflow_password.get_secret_value(),
        settings.approved_dag,
        settings.request_timeout,
    )

    @lru_cache(maxsize=2)
    def kube(write: bool = False):
        from kubernetes import client, config
        from kubernetes.config.config_exception import ConfigException

        configuration = client.Configuration()
        if write:
            if settings.kube_write_config is None:
                raise PermissionError("Restart requires separately configured operator credentials")
            config.load_kube_config(
                config_file=str(settings.kube_write_config),
                context=settings.kube_write_context,
                client_configuration=configuration,
            )
        else:
            try:
                config.load_incluster_config(client_configuration=configuration)
            except ConfigException:
                config.load_kube_config(
                    config_file=str(settings.kubeconfig) if settings.kubeconfig else None,
                    context=settings.kube_context,
                    client_configuration=configuration,
                )
        api_client = client.ApiClient(configuration)
        return KubernetesTools(
            client.CoreV1Api(api_client),
            client.AppsV1Api(api_client),
            settings.kube_namespace,
            settings.deployment_name,
            settings.request_timeout,
        )

    def k(method):
        return lambda **kwargs: getattr(kube(write=method == "restart_deployment"), method)(
            **kwargs
        )

    specs = [
        (
            "dvc_list_datasets",
            "List datasets tracked by DVC in Git.",
            EmptyInput,
            dvc.list_datasets,
        ),
        (
            "dvc_dataset_version",
            "Read dataset pointer, Git commit, and actual working data hash.",
            DatasetInput,
            dvc.dataset_version,
        ),
        (
            "dvc_data_status",
            "Detect changed or missing tracked data using DVC status.",
            EmptyInput,
            dvc.data_status,
        ),
        (
            "dvc_compare_versions",
            "Compare dataset hashes between two Git revisions.",
            CompareInput,
            dvc.compare_versions,
        ),
        (
            "dvc_reproduction_steps",
            "Read a historical dataset pointer at the recorded training Git commit; return reproduction instructions without changing files.",
            ReproduceInput,
            dvc.reproduction_steps,
        ),
        (
            "mlflow_list_experiments",
            "List MLflow experiments.",
            EmptyInput,
            mlflow.list_experiments,
        ),
        (
            "mlflow_list_runs",
            "List recent runs in the Iris experiment.",
            RunsInput,
            mlflow.list_runs,
        ),
        (
            "mlflow_run_details",
            "Retrieve recorded metrics, parameters, artifacts, and dataset provenance for an exact run.",
            RunInput,
            mlflow.run_details,
        ),
        (
            "mlflow_best_run",
            "Compare recent finished runs and select the best recorded metric; default f1_macro.",
            BestInput,
            mlflow.best_run,
        ),
        (
            "mlflow_compare_runs",
            "Compare two or more exact run IDs.",
            CompareRunsInput,
            mlflow.compare_runs,
        ),
        (
            "mlflow_compare_experiments",
            "Compare recent recorded runs across two or more experiment IDs.",
            ExperimentCompareInput,
            mlflow.compare_experiments,
        ),
        (
            "mlflow_registry",
            "Inspect registered models and versions; empty registry is valid.",
            EmptyInput,
            mlflow.registry,
        ),
        ("airflow_list_dags", "List available Airflow 3 DAGs.", EmptyInput, airflow.list_dags),
        ("airflow_dag_runs", "Read recent training DAG run statuses.", DagInput, airflow.dag_runs),
        (
            "airflow_task_instances",
            "Read task statuses and identify failed tasks for an exact DAG run.",
            DagRunInput,
            airflow.task_instances,
        ),
        (
            "airflow_task_logs",
            "Read available logs for a failed task and attempt.",
            TaskLogInput,
            airflow.task_logs,
        ),
        (
            "kubernetes_pods",
            "Inspect prediction API pod readiness, waiting reasons, and restarts in the approved namespace.",
            NamespaceInput,
            k("pods"),
        ),
        (
            "kubernetes_deployments",
            "Inspect desired, ready replicas, image version and model run ID.",
            NamespaceInput,
            k("deployments"),
        ),
        ("kubernetes_services", "Inspect prediction API Services.", NamespaceInput, k("services")),
        (
            "kubernetes_events",
            "Inspect prediction API Kubernetes events for failure diagnostics.",
            NamespaceInput,
            k("events"),
        ),
        (
            "kubernetes_pod_logs",
            "Retrieve bounded prediction API pod logs.",
            PodInput,
            k("pod_logs"),
        ),
    ]
    for name, description, schema, handler in specs:
        registry.add(Tool(name, description, schema, handler))
    registry.add(
        Tool(
            "airflow_trigger_training",
            "State change: trigger the approved training DAG. Requires explicit confirmation outside model conversation.",
            TriggerInput,
            airflow.trigger_training,
            True,
        )
    )
    registry.add(
        Tool(
            "kubernetes_restart_deployment",
            "State change: restart only the approved prediction Deployment. Requires explicit confirmation outside model conversation.",
            RestartInput,
            k("restart_deployment"),
            True,
        )
    )
    return registry
