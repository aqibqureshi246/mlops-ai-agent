from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MLOPS_", env_file=".env", extra="ignore")
    repo_root: Path = Path(".")
    mlflow_tracking_uri: str = "http://localhost:5000"
    experiment_name: str = "iris-classification"
    airflow_url: str = "http://localhost:8080"
    airflow_username: str = "student"
    airflow_password: SecretStr = SecretStr("")
    approved_dag: str = "iris_training"
    kube_namespace: str = "mlops"
    kube_context: str | None = "mlops-agent-reader"
    kubeconfig: Path | None = Path(".runtime/kubeconfig-reader")
    kube_write_config: Path | None = None
    kube_write_context: str | None = "mlops-agent-operator"
    deployment_name: str = "prediction-api"
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_model: str = "qwen2.5:3b"
    agent_api_key: SecretStr = SecretStr("")
    request_timeout: float = Field(default=20, gt=0, le=120)
    llm_timeout: float = Field(default=120, gt=0, le=300)
    max_rounds: int = Field(default=10, ge=1, le=20)
    freshness_days: int = Field(default=30, ge=1)
