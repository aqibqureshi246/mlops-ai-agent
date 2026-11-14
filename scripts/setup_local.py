"""Generate local secrets without printing them or overwriting existing configuration."""

import json
import os
import secrets
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    runtime = ROOT / ".runtime"
    runtime.mkdir(exist_ok=True)
    (ROOT / "artifacts").mkdir(exist_ok=True)
    env = ROOT / ".env"
    if not env.exists():
        password = secrets.token_urlsafe(24)
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        text = text.replace("MLOPS_AIRFLOW_PASSWORD=\n", f"MLOPS_AIRFLOW_PASSWORD={password}\n")
        text = text.replace(
            "MLOPS_AGENT_API_KEY=\n", f"MLOPS_AGENT_API_KEY={secrets.token_urlsafe(32)}\n"
        )
        text += f"\nMLOPS_POSTGRES_PASSWORD={secrets.token_urlsafe(24)}\n"
        text += f"MLOPS_AIRFLOW_JWT_SECRET={secrets.token_urlsafe(48)}\n"
        text += f"MLOPS_AIRFLOW_UID={os.getuid() if hasattr(os, 'getuid') else 50000}\n"
        env.write_text(text, encoding="utf-8")
        if os.name != "nt":
            env.chmod(0o600)
    values = dict(
        line.split("=", 1)
        for line in env.read_text().splitlines()
        if "=" in line and not line.startswith("#")
    )
    target = runtime / "airflow-passwords.json"
    target.write_text(
        json.dumps({values["MLOPS_AIRFLOW_USERNAME"]: values["MLOPS_AIRFLOW_PASSWORD"]}),
        encoding="utf-8",
    )
    if os.name != "nt":
        target.chmod(0o600)
    print("Local configuration ready. Secrets are in ignored files; do not submit them.")


if __name__ == "__main__":
    main()
