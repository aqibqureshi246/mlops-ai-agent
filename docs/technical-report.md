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

The live workflow verified the observations below. CI passed on Python 3.11 and 3.12; the Python 3.12 run recorded 59 passing tests and 86% statement coverage. Actual run IDs, source revision and verification time are recorded at the end of this report.

| Evidence | Export | Status |
|---|---|---|
| Automated tests | evidence/automated-tests.json | 59 passed; 0 skipped; Python 3.12.16 on Linux |
| Local dataset and remote | evidence/local-dvc.json | Hash matches; working and remote DVC status clean |
| DVC reproduction | evidence/dvc-reproduction.json | Isolated checkout and retrieved byte hash verified |
| Local training and prediction | evidence/local-model-verification.json | Actual macro F1 0.933333; quality gate passed; prediction verified |
| Successful training and recovery | evidence/pipeline-none.json | All five recovery tasks succeeded |
| Validation rejection | evidence/pipeline-validation.json | Validator failed; all downstream tasks blocked |
| Quality-gate rejection | evidence/pipeline-quality.json | Evaluation succeeded; final gate rejected the candidate |
| Kubernetes deployment | evidence/kubernetes-deployment.json | Two ready replicas; actual prediction and provenance verified |
| Five conversations | evidence/conversations/01 through 05 JSON files | Verified; see exported evidence |

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
error before preprocessing. Verified outcome: validate fails after its retry; dependent tasks are
upstream_failed, and no approved model is published. Inspect the task log to establish root cause.
Resolution: submit a valid configuration and complete a real recovery run. Actual observations and
run IDs are recorded in evidence/pipeline-validation.json and the recovery export.

### Quality-gate rejection

Injection: trigger iris_training with inject_failure=quality. Evaluation records its measured metrics,
but the final gate explicitly rejects promotion. Verified outcome: validation, preprocessing,
training and evaluation succeeded while the gate failed, quality.passed is false and no approved
bundle exists for that run.
Resolution: remove the injection, rerun and verify the gate and approved model provenance. Do not
lower thresholds simply to pass a failing model. Actual logs are recorded in evidence/pipeline-quality.json.

Both injected failure investigations were executed; their actual task states and root-cause logs
are exported in evidence/. Unit tests independently verify the failure boundaries using fixtures.

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

DVC 3.59.2 failed with an import of the removed private pathspec symbol _DIR_MARK when resolved against pathspec 1.1.1. The dependency is now explicitly pinned to pathspec 0.12.1 in the project and cross-platform lock. Actual dvc add, local remote push, pipeline-graph validation and an isolated worktree remote-pull test subsequently passed. This is an observed dependency failure, distinct from the two deliberately injected Airflow failures above.

## Published-source local verification

The dataset was retrieved into an isolated worktree at commit 98dce180f083d5e64502fb3e5510c1797909a22e and its byte hash matched the DVC pointer. A real local file-backed MLflow training run recorded accuracy 0.9333333333 and macro F1 0.9333333333. Its 0.90 quality gate passed, and the approved bundle returned a successful readiness check, matching application provenance and a setosa prediction through FastAPI TestClient. The exported scope explicitly distinguishes these local checks from the separate real Airflow, Kubernetes and LLM demonstration exports.

## Additional failures found during live verification

The full demonstration exposed issues beyond the deliberately injected training failures. Each
correction was checked in a later live run; the final execution record identifies the source revision
used for the complete demonstration.

| Run | Observed failure | Correction |
|---|---|---|
| [43637611564](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43637611564) | Airflow's SimpleAuthManager opened its password file in append/read mode, but the bind mount was read-only. | Allow the configured Airflow user to write that single ignored password file. |
| [43701309970](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43701309970) | The report regression test imported under python -m pytest but failed under the pytest console entry point. | Load the standalone report module by its explicit file location. |
| [43701421118](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43701421118) | Airflow parsed the DAG, but /auth/token failed because the JWT issuer was not a string. | Configure an explicit deployment issuer and retain redacted DAG-readiness observations. |
| [43702093841](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43702093841) | The separate quality-gate task process tried to resolve proxied MLflow artifacts through its default local tracking location. | Initialize MLflow's tracking URI in that task; a regression test exercises actual remote artifact URI resolution after a process-state reset. |
| [43702642679](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43702642679) | kind v0.26.0 could create the cluster but could not load an image into its containerd 2 node. | Pin kind v0.27.0, kubectl v1.32.2 and the matching Kubernetes node-image digest. The compatibility requirement is documented in the [kind release notes](https://github.com/kubernetes-sigs/kind/releases/tag/v0.27.0). |
| [43703267625](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43703267625) | The small local LLM omitted requested observations and proposed invalid scope arguments. The collector correctly rejected three incomplete conversations. | Supply configured scope and an explicit read-only observation checklist, then give bounded reminders for missing evidence. Tests verify that reminders never execute tools automatically or authorize mutations. |

An additional [3B-model run](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43705224325)
completed dataset, Airflow and Kubernetes conversations but left model-detail and historical-data
observations incomplete. The following experiments used the free local qwen2.5:7b model and limited each
conversation's advertised tools to its declared read-only observation contract. General chat retains
the full tool set. This reduces prompt size without replacing model-selected calls with a script.

The remaining MLflow exceptions also exposed the same proxied-artifact context dependency in the
agent registry: its explicit tracking client was configured, but the global URI used by MLflow's
artifact repository resolver was not. A regression test first reproduced this failure with a real
artifact URI resolver and a stubbed HTTP artifact listing, then passed after registry initialization
set the configured tracking URI. The affected [7B run](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43707447375)
was stopped before rerunning the corrected source. This integration failure is distinct from the
invalid arguments proposed by the small model.

The following [7B execution](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43708905387)
still stopped planning after large MLflow comparison responses, before attempting the dependent
run-details call. The planner context now keeps each comparison candidate's ID, status, time and
metrics while retaining full metadata for the selected candidate. A captured response shrank from
4,086 to 1,792 bytes; a regression test verifies that the full audit evidence is unchanged. Local CPU
inference uses a 300-second request timeout and a bounded 1,024-token planning response. Exported
planner failures include their exception class without disclosing credentials or raw exceptions.

That [CPU run](https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43711970736)
completed model selection, Airflow and Kubernetes conversations, but repeated completed calls and
invalid no-argument DVC calls consumed the job's one-hour limit. Completed required tools are now
removed from subsequent planning rounds. Validation feedback lists rejected field names and issue
types without echoing argument values; literal identifier hints are copied only from successful
observations. The final demonstration returns to the faster local qwen2.5:3b model.

Dataset-status reads now target Git-tracked DVC pointer files, rather than unrelated missing outputs
from the optional local DVC training graph. A real DVC test verifies both a clean tracked dataset
beside an unmaterialized pipeline and detection of changed dataset bytes. Airflow pipeline execution
continues to be verified independently through its actual task states and logs.

The collector records its required successful tools and model name in each conversation export.
The LLM still selects every tool and argument; the registry executes actual adapter calls. If the
model stops early, cannot access a service or reaches its round limit, missing evidence remains
explicit and full-verification publication fails. A checklist is not a replacement for observations.

## Cost and lifetime of the demonstration

The verification uses a standard Linux runner for this public GitHub repository, local Docker
services, an ephemeral kind cluster and the local qwen2.5:3b model through Ollama. The hosted-model
workflow input is disabled. No paid hosting or hosted LLM service is needed. Services and the cluster
are torn down after the run; exported observations are the submission evidence, not a permanently
hosted prediction endpoint.

GitHub documents standard hosted runners as free for public repositories in its
[runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).


## Verified live execution record

- verified_at: 2026-11-15T15:20:28.943657+00:00
- recovery_run: manual__agent_c1943827aa264a279ed2d833d8c1ec01
- validation_failure_run: manual__agent_03911bfcbe8646e4b33d511d32391e9f
- quality_failure_run: manual__agent_1ba55e0bddd442e48a0d2944ed9d0084
- dataset_hash: 4224576f0267bf88902f87f0f6200967
- complete_conversations: 5
- tests_passed: 59
- test_python: 3.12.16
- test_platform: Linux
- workflow_url: https://github.com/aqibqureshi246/mlops-ai-agent/actions/runs/43727102101
- source_commit: 64673783c3e301569b7d04665dccc6afee1755a6

All five recovery tasks succeeded. The validation failure log identified the injected missing-feature error; the quality failure log identified the explicit gate rejection. Both were followed by a successful recovery run. Historical data bytes were retrieved and verified, and the deployment/inference observations are exported in evidence/.
