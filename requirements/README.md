# Dependency environments

The agent and local training environment use the cross-platform uv.lock at the repository root.
The containers keep server dependencies separate so Airflow's database and provider requirements
do not constrain the MLflow tracking server, or vice versa.

- airflow.txt is the Airflow image input; airflow-lock.txt resolves Linux Python 3.12 dependencies and hashes.
- mlflow.txt is the tracking server input; mlflow-lock.txt pins its compatible SQLAlchemy runtime and hashes.
- inference.txt pins the prediction runtime; constraints.txt supplies the client environment's resolved versions.

Regenerate the server locks with the pinned uv version from the repository root:

~~~text
uv pip compile requirements/airflow.txt --constraint requirements/constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes --output-file requirements/airflow-lock.txt
uv pip compile requirements/mlflow.txt --constraint requirements/constraints.txt --python-version 3.12 --python-platform x86_64-unknown-linux-gnu --generate-hashes --output-file requirements/mlflow-lock.txt
~~~

The images install with hash verification and run pip check. Import checks also verify the actual
Airflow SDK and MLflow SQL backend before service startup. Airflow uses the official slim image,
avoiding unrelated cloud-provider integrations that the training DAG does not need.

Airflow provider versions follow its release-tested constraints:
https://raw.githubusercontent.com/apache/airflow/constraints-3.0.1/constraints-3.12.txt
