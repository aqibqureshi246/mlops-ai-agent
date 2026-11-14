"""Create short-lived, namespace-restricted agent credentials without printing tokens."""

import argparse
import json
import os
import subprocess
from pathlib import Path

import yaml
from dotenv import set_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", default="kind-mlops-a3")
    parser.add_argument("--include-operator", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    if not args.confirm:
        parser.error("--confirm is required to issue short-lived ServiceAccount credentials")
    raw = subprocess.check_output(
        [
            "kubectl",
            "--context",
            args.context,
            "config",
            "view",
            "--minify",
            "--flatten",
            "--raw",
            "-o",
            "json",
        ],
        text=True,
    )
    config = json.loads(raw)
    cluster = config["clusters"][0]
    runtime = Path(".runtime")
    runtime.mkdir(exist_ok=True)
    roles = ["reader", "operator"] if args.include_operator else ["reader"]
    for role in roles:
        name = f"mlops-agent-{role}"
        token = subprocess.check_output(
            [
                "kubectl",
                "--context",
                args.context,
                "-n",
                "mlops",
                "create",
                "token",
                name,
                "--duration=3600s",
            ],
            text=True,
        ).strip()
        scoped = {
            "apiVersion": "v1",
            "kind": "Config",
            "clusters": [cluster],
            "users": [{"name": name, "user": {"token": token}}],
            "contexts": [
                {
                    "name": name,
                    "context": {"cluster": cluster["name"], "user": name, "namespace": "mlops"},
                }
            ],
            "current-context": name,
        }
        path = runtime / f"kubeconfig-{role}"
        path.write_text(yaml.safe_dump(scoped))
        if os.name != "nt":
            path.chmod(0o600)
        if role == "operator":
            set_key(Path(".env"), "MLOPS_KUBE_WRITE_CONFIG", str(path))
    print("Issued short-lived scoped credentials into ignored local files. No tokens were printed.")


if __name__ == "__main__":
    main()
