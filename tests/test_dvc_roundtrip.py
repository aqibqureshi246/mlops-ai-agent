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
