"""Materialize the public Iris dataset and populate a local DVC remote."""

import argparse
import subprocess
from pathlib import Path

from sklearn.datasets import load_iris

ROOT = Path(__file__).resolve().parents[1]


def command(*argv):
    subprocess.run(argv, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", type=Path, default=ROOT / ".runtime" / "dvc-remote")
    args = parser.parse_args()
    target = ROOT / "data" / "iris.csv"
    target.parent.mkdir(exist_ok=True)
    iris = load_iris(as_frame=True).frame
    iris.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width", "target"]
    iris.to_csv(target, index=False, lineterminator="\n")
    if not (ROOT / ".dvc").exists():
        command("dvc", "init")
    pointer = ROOT / "data/iris.csv.dvc"
    if pointer.exists():
        import hashlib

        import yaml

        recorded = yaml.safe_load(pointer.read_text())["outs"][0]["md5"]
        actual = hashlib.md5(target.read_bytes(), usedforsecurity=False).hexdigest()
        if recorded != actual:
            raise ValueError(
                "Regenerated data disagrees with committed pointer; do not overwrite provenance"
            )
        # A fresh clone has the pointer but no content-addressed cache yet.
        command("dvc", "commit", "data/iris.csv.dvc")
    else:
        command("dvc", "add", "data/iris.csv")
    args.remote.resolve().mkdir(parents=True, exist_ok=True)
    command(
        "dvc", "remote", "add", "--local", "--force", "-d", "storage", str(args.remote.resolve())
    )
    command("dvc", "push", "data/iris.csv.dvc")
    print("Dataset matches its DVC pointer and has been pushed to the local remote.")


if __name__ == "__main__":
    main()
