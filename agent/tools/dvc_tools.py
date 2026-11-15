import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml
from pydantic import Field

from agent.schemas import StrictInput


class DatasetInput(StrictInput):
    path: str = "data/iris.csv.dvc"


class CompareInput(StrictInput):
    old_rev: str = Field(min_length=1, max_length=100)
    new_rev: str = "HEAD"


class ReproduceInput(DatasetInput):
    revision: str = "HEAD"


class DvcTools:
    def __init__(self, root: Path, timeout: float = 20):
        self.root = root.resolve()
        self.timeout = timeout

    def _run(self, *args: str) -> str:
        # Only internal code builds argv; shell syntax is never interpreted.
        result = subprocess.run(
            args,
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
            timeout=self.timeout,
            encoding="utf-8",
        )
        return result.stdout.strip()

    def _revision(self, revision: str) -> str:
        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._/~-]{0,99}", revision):
            raise ValueError("Invalid Git revision")
        return self._run("git", "rev-parse", "--verify", f"{revision}^{{commit}}")

    def _pointer(self, path: str) -> Path:
        candidate = (self.root / path).resolve()
        if not candidate.is_relative_to(self.root) or candidate.suffix != ".dvc":
            raise ValueError("Dataset must be a .dvc pointer inside the repository")
        if ".git" in candidate.relative_to(self.root).parts:
            raise ValueError("Git internals are not datasets")
        return candidate

    def list_datasets(self) -> dict:
        tracked = self._run("git", "ls-files", "--", "*.dvc").splitlines()
        return {"datasets": [p for p in tracked if p.endswith(".dvc")]}

    def dataset_version(self, path: str = "data/iris.csv.dvc") -> dict:
        pointer = self._pointer(path)
        output = yaml.safe_load(pointer.read_text(encoding="utf-8"))["outs"][0]
        target = (pointer.parent / output["path"]).resolve()
        if not target.is_relative_to(self.root):
            raise ValueError("Tracked data escapes repository")
        exists = target.is_file()
        # DVC 3 file pointers use MD5; reading binary avoids platform newline changes.
        actual = (
            hashlib.md5(target.read_bytes(), usedforsecurity=False).hexdigest() if exists else None
        )
        return {
            "pointer": path,
            "git_commit": self._revision("HEAD"),
            "recorded_hash": output.get("md5"),
            "working_hash": actual,
            "data_exists": exists,
            "matches_pointer": exists and actual == output.get("md5"),
            "pointer_dirty": bool(self._run("git", "status", "--porcelain", "--", path)),
        }

    def data_status(self) -> dict:
        pointers = self.list_datasets()["datasets"]
        changes = json.loads(self._run("dvc", "status", "--json", *pointers)) if pointers else {}
        return {"changes": changes}

    def compare_versions(self, old_rev: str, new_rev: str = "HEAD") -> dict:
        old, new = self._revision(old_rev), self._revision(new_rev)
        return {
            "old_commit": old,
            "new_commit": new,
            "diff": json.loads(self._run("dvc", "diff", "--json", old, new)),
        }

    def reproduction_steps(self, revision: str = "HEAD", path: str = "data/iris.csv.dvc") -> dict:
        self._pointer(path)
        commit = self._revision(revision)
        pointer_at_commit = self._run("git", "show", f"{commit}:{path}")
        recorded = yaml.safe_load(pointer_at_commit)["outs"][0]
        return {
            "git_commit": commit,
            "dataset": path,
            "recorded_hash": recorded.get("md5"),
            "steps": [
                f"git worktree add --detach ../mlops-reproduce {commit}",
                "cd ../mlops-reproduce",
                "python -m venv .venv",
                "Activate the new .venv for your shell",
                "python -m pip install uv==0.7.3",
                "uv sync --locked",
                "dvc remote add --local -d storage <absolute-path-to-existing-remote>",
                f"dvc pull {path}",
                "dvc status",
            ],
            "note": "Instructions only. Requires access to the DVC remote; current checkout is untouched.",
        }
