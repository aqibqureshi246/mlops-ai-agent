"""Demonstrate an isolated DVC checkout and verify the actual retrieved bytes."""

import hashlib
import json
import subprocess
from pathlib import Path
from uuid import uuid4

import yaml

ROOT = Path(__file__).resolve().parents[1]


def command(*args, cwd=ROOT):
    return subprocess.check_output(args, cwd=cwd, text=True).strip()


def main():
    commit = command("git", "rev-parse", "HEAD")
    runtime = (ROOT / ".runtime").resolve()
    runtime.mkdir(exist_ok=True)
    worktree = (runtime / f"reproduce-{uuid4().hex[:10]}").resolve()
    if not worktree.is_relative_to(runtime):
        raise ValueError("Reproduction directory must remain inside local runtime storage")
    command("git", "worktree", "add", "--detach", str(worktree), commit)
    try:
        remote = runtime / "dvc-remote"
        command("dvc", "remote", "add", "--local", "-d", "storage", str(remote), cwd=worktree)
        command("dvc", "pull", "data/iris.csv.dvc", cwd=worktree)
        pointer = yaml.safe_load((worktree / "data/iris.csv.dvc").read_text())["outs"][0]
        actual = hashlib.md5(
            (worktree / "data/iris.csv").read_bytes(), usedforsecurity=False
        ).hexdigest()
        if actual != pointer["md5"]:
            raise ValueError("Retrieved dataset disagrees with historical pointer")
        Path("evidence/dvc-reproduction.json").write_text(
            json.dumps(
                {
                    "git_commit": commit,
                    "pointer": "data/iris.csv.dvc",
                    "recorded_hash": pointer["md5"],
                    "retrieved_hash": actual,
                    "matches": True,
                    "method": "Isolated detached Git worktree, dvc pull, byte-hash verification",
                },
                indent=2,
            )
            + "\n"
        )
        print("Historical dataset retrieved and hash verified in an isolated worktree.")
    finally:
        # The absolute target was checked against the dedicated runtime directory above.
        command("git", "worktree", "remove", "--force", str(worktree))


if __name__ == "__main__":
    main()
