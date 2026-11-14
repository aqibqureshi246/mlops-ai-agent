"""Build an explicitly approved local model bundle and deploy to the named kind cluster."""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def command(*args):
    subprocess.run(args, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--cluster", default="mlops-a3")
    parser.add_argument(
        "--confirm", action="store_true", help="Explicitly authorize this local cluster deployment"
    )
    args = parser.parse_args()
    if not args.confirm:
        parser.error(
            "Review the manifests and provide --confirm to create/update the local namespace"
        )
    bundle = args.bundle.resolve()
    metadata = json.loads((bundle / "metadata.json").read_text())
    if not metadata.get("quality_passed"):
        raise ValueError("Only quality-approved bundles can be deployed")
    context = ROOT / ".runtime" / "build-context"
    context.mkdir(parents=True, exist_ok=True)
    for name in ["api", "requirements"]:
        shutil.copytree(
            ROOT / name,
            context / name,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__"),
        )
    shutil.copytree(bundle, context / "model", dirs_exist_ok=True)
    shutil.copy2(ROOT / "Dockerfile", context / "Dockerfile")
    image = f"mlops-prediction:{metadata['run_id'][:12]}"
    command("docker", "build", "-t", image, str(context))
    command("kind", "load", "docker-image", image, "--name", args.cluster)
    context_name = f"kind-{args.cluster}"
    for name in ["namespace", "configmap", "service", "rbac"]:
        command("kubectl", "--context", context_name, "apply", "-f", f"kubernetes/{name}.yaml")
    manifest = yaml.safe_load((ROOT / "kubernetes/deployment.yaml").read_text())
    manifest["spec"]["template"]["spec"]["containers"][0]["image"] = image
    manifest["spec"]["template"]["metadata"]["annotations"] = {
        "mlops/model-run-id": metadata["run_id"]
    }
    rendered = ROOT / ".runtime" / "deployment.yaml"
    rendered.write_text(yaml.safe_dump(manifest))
    command("kubectl", "--context", context_name, "apply", "-f", str(rendered))
    command(
        "kubectl",
        "--context",
        context_name,
        "-n",
        "mlops",
        "rollout",
        "status",
        "deployment/prediction-api",
        "--timeout=180s",
    )


if __name__ == "__main__":
    main()
