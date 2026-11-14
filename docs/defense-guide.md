# Engineering defense guide

## Explain the system in one minute

An LLM receives an operations request and chooses from typed tools. Each tool talks to the actual
DVC repository, MLflow tracking API, Airflow 3 API or Kubernetes API. The registry validates inputs,
restricts operations and records an audit log. Successful calls return timestamped evidence;
failures remain unknown. The final report uses those observations and an explicit policy rather
than trusting generated operational claims. Training and cluster changes need human confirmation.

## Be ready to demonstrate

| Scenario | What to show | Why it matters |
|---|---|---|
| Dataset provenance | MLflow run's data.hash and data.git_commit; DVC pointer at that commit | Current HEAD alone cannot establish historical training data |
| Model selection | Five recorded runs, macro F1, parameters, artifacts and quality tag | Highest F1 is a candidate ranking, not a deployment approval |
| Pipeline execution | Real Airflow run and five task states, including retries | A successful POST only means the run was accepted |
| Deployment diagnosis | Desired/ready replicas, Pod Ready condition, restarts, events, logs | Running phase alone does not imply service readiness |
| Multi-tool reasoning | Selected run -> historical DVC hash -> deployed run -> workflow state | Training, data and infrastructure are different sources of evidence |

## Likely questions

**Why direct tool calling?** The task needs a small, reviewable control loop with typed inputs and
outputs. Provider-compatible tool calling supports multiple models without hiding permission checks
inside a framework. The planner selects tools; the registry controls execution.

**How do you avoid data leakage?** The stratified split is seeded and stored. StandardScaler is
inside the sklearn pipeline and fits only training rows. Evaluation uses held-out rows. The small
single holdout is a limitation; cross-validation and independent validation are future improvements.

**When is retraining justified?** The configured freshness policy can identify an old candidate.
Failed quality gates require investigation. No live degradation metrics are collected, so deployment
failure or a newer dataset alone cannot prove model drift. A status question never starts training.

**What happens if a service is down?** Its tool returns error/timeout, empty operational data and an
uncertainty. Other service results remain usable. Nothing reports healthy state without observations.

**Why two probes?** Liveness asks whether the process is alive. Readiness asks whether a verified
approved model is loaded. A process without a model stays alive for diagnostics but receives no
Service traffic because readiness returns HTTP 503.

**How is a change authorized?** Exact validated arguments are stored server-side in a short-lived,
one-use action. An authenticated user confirms through a separate interface. The LLM has no
confirmation tool and cannot add authorization fields. Namespace and application checks run again
at the adapter, and RBAC supplies an independent infrastructure boundary.

**Why version the dataset and code separately?** Git records source and a small DVC pointer. DVC
stores content-addressed data outside Git. MLflow records both the source commit and dataset hash,
allowing a previous run to be reconstructed with an isolated Git worktree and the appropriate remote.

**What is still needed for production?** External identity management, TLS, persistent confirmation
storage, independent validation and drift metrics, a shared object-store DVC remote, HA Airflow,
managed tracking storage and artifact storage, image signing and stronger supply-chain controls.
The submitted local deployment demonstrates the required engineering behavior without claiming HA.

## Unexpected failure approach

Read the first failed task or unhealthy Pod condition. Inspect its actual logs and events. Check
credentials and connectivity without printing secrets. Verify dataset and image provenance. Fix the
root cause, rerun the affected workflow or deploy a verified bundle, and observe terminal task states
or a healthy rollout before declaring recovery. Keep the original failure and recovery evidence.
