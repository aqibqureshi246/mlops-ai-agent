# Technical report: MLOps AI Agent

## Objective and architecture

The system exposes one conversational interface across DVC, MLflow, Apache Airflow 3 and Kubernetes.
The LLM selects typed tools; a separate registry enforces validation, permissions and explicit change
confirmation. Returned values become timestamped evidence. Operational reporting is rendered from
that evidence, with a deterministic assessment of freshness, provenance and deployment readiness.
The complete topology, data flow and trust boundaries are in [architecture.md](../architecture.md).

## Tool definitions and permissions

| Module | Callable tools | Permission |
|---|---|---|
| DVC | list_datasets, dataset_version, data_status, compare_versions, reproduction_steps | Read local Git/DVC metadata and data; historical reproduction is instructions only |
| MLflow | list_experiments, list_runs, run_details, best_run, compare_runs, compare_experiments, registry | Read tracking and registry metadata |
| Airflow | list_dags, dag_runs, task_instances, task_logs | Authenticated API v2 reads |
| Airflow | trigger_training | Exact allowlisted DAG; explicit confirmation |
| Kubernetes | pods, deployments, services, events, pod_logs | Approved namespace and application; bounded logs |
| Kubernetes | restart_deployment | Approved namespace and Deployment; explicit confirmation and operator RBAC |

All schemas reject unknown fields. Path traversal, shell-style revisions, unsupported DAGs and
out-of-scope Kubernetes objects are rejected. Confirmations expire, are bound to the caller and
cannot be replayed. The planner cannot self-authorize. Structured logs contain request ID, tool,
status and duration without raw payloads.

## Dataset, training and deployment

The canonical 150-row Iris CSV is managed by DVC. Validation checks schema, data completeness,
class representation, positive finite features, pointer hash and clean tracked source. A seeded
stratified 80/20 split isolates evaluation data; a scaling/logistic regression pipeline fits training
rows only. MLflow records macro F1, accuracy, hyperparameters, dataset hash, Git commit, pointer,
Airflow run ID and artifacts. The final gate publishes an approved bundle only if macro F1 meets
the configured threshold, normally 0.90.

Airflow implements validate -> preprocess -> train -> evaluate_mlflow -> quality_gate with one retry
per task and all-success dependencies. Artifacts are isolated by DAG run. The inference image carries
the approved model, SHA-256 and provenance. Kubernetes deploys two replicas with a ClusterIP Service,
ConfigMap, probes, resource requests/limits and restricted container security settings.

## Validation record

Local verification completed with 43 passing tests and 85% statement coverage. Lint, formatting and Compose configuration checks passed. DVC versioning, local remote storage and isolated worktree recovery were exercised. Live Airflow, Kubernetes and LLM conversations remain pending because Docker Desktop is unavailable on the current approximately 4 GB host. The manual GitHub workflow is provided for complete live verification; hosted runs are currently awaiting runner availability. No unexecuted demonstration is represented as successful.

| Evidence | Expected export | Status |
|---|---|---|
| Automated tests | evidence/automated-tests.json | 43 passed locally; two dependency deprecation warnings |
| Local dataset and remote | evidence/local-dvc.json | Hash matches; working and remote DVC status clean |
| DVC reproduction | evidence/dvc-reproduction.json | Verified locally against published Git commit |
| Local training and prediction | evidence/local-model-verification.json | Actual macro F1 0.933333; quality gate passed; prediction verified |
| Successful training and recovery | evidence/pipeline-none.json | Pending live execution |
| Validation rejection | evidence/pipeline-validation.json | Pending live execution |
| Quality-gate rejection | evidence/pipeline-quality.json | Pending live execution |
| Kubernetes deployment | evidence/kubernetes-deployment.json | Pending live execution |
| Five conversations | evidence/conversations/01 through 05 JSON files | Pending live execution |

## Five required conversations

The exact requests and required tool evidence are defined in scripts/collect_demo.py:

1. Dataset provenance: current DVC pointer, Git commit, byte hash and changed-data status.
2. Model selection: comparison of five recent runs, selected candidate and detailed recorded artifacts.
3. Pipeline execution: recent real Airflow DAG runs and task statuses after an explicitly authorized run.
4. Deployment diagnosis: Deployment, Pod readiness, Service and event observations.
5. Multi-tool reasoning: selected MLflow run, historical DVC reference, Kubernetes status and Airflow history.

Each JSON export stores the actual request, evidence envelopes, report, uncertainties and recommendations.
The collector marks a conversation incomplete and exits unsuccessfully if required observations are
missing. An illustrative transcript or mocked CI result does not count as live demonstration evidence.

## Deliberately injected failure investigations

### Validation rejection

Injection: trigger iris_training with inject_failure=validation. The validator raises a missing-feature
error before preprocessing. Expected diagnosis: validate fails after its retry; dependent tasks are
upstream_failed, and no approved model is published. Inspect the task log to establish root cause.
Resolution: submit a valid configuration and complete a real recovery run. Actual observations and
run IDs belong in evidence/pipeline-validation.json and the recovery export.

### Quality-gate rejection

Injection: trigger iris_training with inject_failure=quality. Evaluation records its measured metrics,
but the final gate explicitly rejects promotion. Expected diagnosis: training and evaluation can
succeed while the gate fails, quality.passed is false and no approved bundle exists for that run.
Resolution: remove the injection, rerun and verify the gate and approved model provenance. Do not
lower thresholds simply to pass a failing model. Actual logs belong in evidence/pipeline-quality.json.

These are planned reproducible investigations until the corresponding real exports exist. Unit tests
independently verify the failure boundaries using controlled fixtures.

## Limitations and improvements

The small public dataset and single test split are educational and do not support claims of deployment
suitability in a real domain. There is no live drift detector. An old model may violate the configured
freshness policy, but deployment failure alone does not justify retraining. The local Compose stack
uses development authentication and a single MLflow server; production needs TLS, external identities,
authenticated tracking, HA services, shared object storage and durable confirmation storage.

The kind administrative context is used only for installation. The agent defaults to a separately
issued read-only ServiceAccount token; confirmed restarts require separate operator credentials. SHA-256 detects accidental artifact corruption but does
not turn arbitrary pickle files into safe input. Small local LLMs can omit tools; missing observations
remain uncertainties. Error classifications and bounded retries can be extended as operational needs
grow. Direct dependencies are pinned and a complete cross-platform dependency lock is included.

## Unexpected failure resolved during verification

DVC 3.59.2 failed with an import of the removed private pathspec symbol _DIR_MARK when resolved against pathspec 1.1.1. The dependency is now explicitly pinned to pathspec 0.12.1 in the project and cross-platform lock. Actual dvc add, local remote push, pipeline-graph validation and an isolated worktree remote-pull test subsequently passed. This is an observed dependency failure, distinct from the two planned deliberately injected Airflow failures above.

## Published-source local verification

The dataset was retrieved into an isolated worktree at commit 98dce180f083d5e64502fb3e5510c1797909a22e and its byte hash matched the DVC pointer. A real local file-backed MLflow training run recorded accuracy 0.9333333333 and macro F1 0.9333333333. Its 0.90 quality gate passed, and the approved bundle returned a successful readiness check, matching application provenance and a setosa prediction through FastAPI TestClient. The exported scope explicitly distinguishes these local checks from pending real Airflow, Kubernetes and LLM demonstrations.
