# MLOps AI Agent

A Python agent that gathers operational evidence from DVC, MLflow, Apache Airflow 3 and Kubernetes.
It uses an OpenAI-compatible LLM to select typed tools, then reports actual tool results and an
explicit retraining policy. State-changing requests require a separate human confirmation.

## What the system does

| Integration | Capabilities |
|---|---|
| DVC | Tracked datasets, current and historical hashes, Git provenance, changes and reproduction steps |
| MLflow | Experiments, recent runs, metric comparison, artifacts and optional model registry |
| Airflow 3 | Authenticated API v2, DAG runs, task failures, logs and confirmed training triggers |
| Kubernetes | Pods, Deployments, Services, events, bounded logs and confirmed scoped restarts |

See [architecture.md](architecture.md), the [technical report](docs/technical-report.md),
and the [engineering defense guide](docs/defense-guide.md).

## Requirements

- Python 3.11 or 3.12, Git, Docker Desktop with Linux containers and Docker Compose.
- kubectl and kind for the local Kubernetes demonstration.
- An OpenAI-compatible model with tool calling. A local Ollama profile is provided; alternatively
  configure a hosted model in the ignored .env file. Never put keys in Git or chat transcripts.
- Allow sufficient memory and disk space for Airflow, PostgreSQL, MLflow, kind and a local LLM.
  Airflow runs in Linux containers because native Windows execution is unsupported.

## 1. Install and generate local configuration

Run in the repository root. PowerShell:

~~~powershell
python -m pip install uv==0.7.3
uv sync --locked --extra dev
.venv\Scripts\Activate.ps1
python scripts/setup_local.py
~~~

Linux/macOS: activate with source .venv/bin/activate; other Python commands are identical.
Dependencies are pinned in pyproject.toml and resolved across supported platforms in uv.lock.
Service-specific requirements are in requirements/.
Local passwords and the agent HTTP key are generated into ignored files and are never printed.

## 2. Materialize and version the dataset

~~~text
python scripts/prepare_data.py
~~~

The dataset is the 150-row Iris dataset distributed with scikit-learn. The script regenerates its
canonical CSV, verifies the committed DVC hash, and uploads it into an ignored local DVC remote.
After a fresh clone this recreates the public data without needing access to someone else's remote.
For an existing external remote, configure its URL locally and use dvc pull data/iris.csv.dvc instead.

Commit tracked source and pointer changes before training: training rejects dirty tracked files.
The binary dataset and model artifacts are managed by DVC or runtime storage, not committed to Git.

## 3. Launch MLflow and Airflow

~~~text
docker compose up -d --build
~~~

- MLflow UI: http://localhost:5000
- Airflow UI: http://localhost:8080; username student, password from local .env.
- Airflow authenticates API calls using a JWT obtained from /auth/token, then uses /api/v2.
- Airflow's five dependent tasks are validate, preprocess, train, evaluate_mlflow, quality_gate.
  Each has one retry; validation and quality failures prevent an approved bundle being published.
- Compose uses PostgreSQL for Airflow and persistent volumes for metadata, artifacts and logs.
  Airflow standalone and MLflow SQLite are local demonstration choices, not an HA production setup.

## 4. Choose an LLM

For local Ollama:

~~~text
docker compose --profile local-llm up -d ollama
docker compose exec ollama ollama pull qwen2.5:3b
~~~

The defaults target http://localhost:11434/v1 with model qwen2.5:3b. A small local model can make
planning mistakes; missing evidence remains unknown. A stronger tool-calling model can be configured
with MLOPS_LLM_BASE_URL, MLOPS_LLM_MODEL and MLOPS_LLM_API_KEY in .env.
A provider outage never falls back to invented results or a scripted demonstration planner.

## 5. Run real training and failure investigations

~~~text
python scripts/run_pipeline.py --confirm
python scripts/run_pipeline.py --inject-failure validation --confirm
python scripts/run_pipeline.py --inject-failure quality --confirm
python scripts/run_pipeline.py --confirm
~~~

The --confirm flag explicitly authorizes each state-changing training request. The script polls
actual task states, exports available failure logs and rejects unexpected outcomes. JSON exports
appear in evidence/. Inspect the first failed task and the gate rejection before rerunning.
The last successful run demonstrates recovery. Approved bundles are under artifacts/<run-hash>/approved.

For a second comparison candidate outside Airflow:

~~~text
python -m training.train all --c 0.3 --work artifacts/candidate-03
~~~

The DVC pipeline can also execute the same stages with dvc repro while MLflow is running.
Airflow remains the required orchestration demonstration; a local run is not a substitute.

## 6. Deploy the prediction service on Kubernetes

~~~text
kind create cluster --name mlops-a3
python scripts/deploy.py --bundle artifacts/<successful-run-hash>/approved --confirm
python scripts/configure_kube_access.py --confirm
kubectl --context kind-mlops-a3 -n mlops get deployments,pods,services
kubectl --context kind-mlops-a3 -n mlops port-forward service/prediction-api 8000:80
~~~

Deployment uses two replicas, a ClusterIP Service, readiness/liveness/startup probes, resource
requests and limits, and a ConfigMap. The image and Pod annotation identify the exact MLflow run.
The image bundles only a quality-approved model and verifies its SHA-256 before loading it.
Open http://localhost:8000/docs and send four positive Iris measurements to POST /predict.
GET /version returns the deployed run and dataset provenance.

Read-only and narrowly scoped operator ServiceAccounts are in kubernetes/rbac.yaml. The installer uses the kind administration context; the agent uses a separate read-only
ServiceAccount kubeconfig. Credentials expire after one hour and can be renewed by rerunning
configure_kube_access.py. For a confirmed restart demonstration, explicitly run that script with
--include-operator --confirm; this configures separate operator credentials for restart calls only.
Do not commit kubeconfigs or ServiceAccount tokens.

## 7. Interact with the agent

~~~text
mlops-agent "Compare the last five training runs by f1_macro."
mlops-agent "Why is my prediction API unavailable? Check pods, events and logs."
mlops-agent "Find the best recent model, identify the dataset version used to train it, check the deployment status, and explain whether retraining is necessary."
mlops-agent --action airflow_trigger_training
mlops-agent --action kubernetes_restart_deployment
~~~

The action commands display exact parameters and require typing CONFIRM. Chat requests can propose
an action but cannot execute it. HTTP users use a one-use, five-minute action confirmation token
bound to their authenticated identity. Supplying confirmed=true to an LLM tool is rejected.

Optional authenticated agent HTTP API:

~~~text
uvicorn agent.server:app --host 127.0.0.1 --port 9000
~~~

Send X-API-Key using the generated MLOPS_AGENT_API_KEY. POST /chat takes a message.
POST /actions/propose takes tool and arguments. POST /actions/confirm takes token and confirm.
Keep the server on loopback unless adding TLS, identity management and appropriate access controls.

## 8. Test and collect submission evidence

~~~text
ruff check .
ruff format --check .
pytest
python scripts/collect_demo.py
python scripts/check_submission.py
~~~

Ordinary CI uses mocked service clients. Tests also train a real small sklearn model in temporary
storage to verify prediction and quality-gate behavior. They do not claim live cluster validation.
GitHub Actions runs lint, formatting and tests on pull requests and main pushes for Python 3.11/3.12.

The evidence collector runs five real LLM conversations and fails if required successful tool
responses are missing. Exported evidence is acceptable in place of screenshots. See the report for
which checks and live demonstrations have actually been completed.

## Live demonstration on GitHub

A machine with approximately 4 GB RAM cannot reliably run all services and a local LLM together.
The manual Live demonstration workflow runs the complete sequence on a Linux GitHub runner.
Open Actions, choose Live demonstration and run it on main. It installs the locked environment,
executes real training and failure scenarios, deploys to an ephemeral kind cluster, tests prediction
and captures the five conversations. It stops if required evidence is missing. After a successful
run it can commit sanitized evidence to main; this is controlled by the publish_evidence input.

For a hosted model, add the LLM_API_KEY Actions secret and LLM_BASE_URL and LLM_MODEL repository
variables and explicitly enable the use_hosted_llm workflow input. The key must remain in the secret
store. The workflow defaults to free local Ollama with qwen2.5:7b even when a hosted key is configured. Model quality,
runner resources and external availability can affect a live run.
The workflow does not claim success until the observations are complete.

## Submission contents

Submit the GitHub repository URL. Relevant tracked files are source, tests, manifests, pipeline and
service configuration, the DVC pointer, documentation and sanitized exported evidence. The original
assignment PDF, credentials, virtual environments, generated models, databases, caches and temporary
build files are ignored. Review git diff --cached and git status before publishing.

## Known limits

The demonstration uses a small public dataset and a single holdout; its high F1 does not establish
production readiness. No live drift detector is configured. The freshness policy is 30 days and
retraining is only recommended with evidence, never automatically triggered by a status query.
Confirmations are process-local; restarting the HTTP process invalidates outstanding tokens.
Remote service outages and incomplete LLM plans remain explicit uncertainties.
