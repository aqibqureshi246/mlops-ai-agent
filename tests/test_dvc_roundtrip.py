import os
import subprocess


def test_real_dvc_remote_pull_restores_data_in_isolated_worktree(tmp_path):
    source, clone, remote = [tmp_path / name for name in ("source", "clone", "remote")]
    source.mkdir()
    remote.mkdir()
    env = {**os.environ, "DVC_NO_ANALYTICS": "1"}

    def run(*args, cwd=source):
        subprocess.run(
            args, cwd=cwd, env=env, check=True, capture_output=True, text=True, timeout=30
        )

    run("git", "init", "-q")
    run("dvc", "init", "-q")
    (source / "sample.csv").write_bytes(b"feature,target\n1,0\n")
    run("dvc", "add", "sample.csv", "-q")
    run("dvc", "remote", "add", "-d", "storage", str(remote))
    run("dvc", "push", "sample.csv.dvc", "-q")
    run("git", "add", ".")
    run(
        "git",
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-qm",
        "data",
    )
    run("git", "worktree", "add", "--detach", str(clone), "HEAD")
    assert not (clone / "sample.csv").exists()
    run("dvc", "pull", "sample.csv.dvc", "-q", cwd=clone)
    assert (clone / "sample.csv").read_bytes() == b"feature,target\n1,0\n"


def test_bootstrap_with_existing_pointer_populates_the_remote(tmp_path, monkeypatch):
    import hashlib
    import importlib.util
    import shutil
    import sys
    from pathlib import Path

    import yaml

    project = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "prepare_data", project / "scripts/prepare_data.py"
    )
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "data").mkdir()
    shutil.copy2(project / "data/iris.csv.dvc", tmp_path / "data/iris.csv.dvc")
    monkeypatch.setattr(bootstrap, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["prepare_data.py"])
    monkeypatch.setenv("DVC_NO_ANALYTICS", "1")
    bootstrap.main()
    digest = yaml.safe_load((tmp_path / "data/iris.csv.dvc").read_text())["outs"][0]["md5"]
    stored = tmp_path / ".runtime/dvc-remote/files/md5" / digest[:2] / digest[2:]
    assert stored.is_file()
    assert hashlib.md5(stored.read_bytes(), usedforsecurity=False).hexdigest() == digest
