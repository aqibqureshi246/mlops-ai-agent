"""Check tracked file selection and common credential patterns before publication."""

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = {".env", "config.local", "kubeconfig", "credentials", "hosts.yml"}
PATTERNS = [
    r"gh[pousr]_[A-Za-z0-9]{30,}",
    r"sk-[A-Za-z0-9]{32,}",
    r"AKIA[0-9A-Z]{16}",
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
]


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    errors = []
    for name in filter(None, paths):
        path = Path(name)
        if path.name in FORBIDDEN or any(
            p in path.parts for p in (".venv", ".tmp", ".runtime", "__pycache__", "artifacts")
        ):
            errors.append(f"Irrelevant or sensitive tracked path: {name}")
        if (ROOT / name).is_file():
            text = (ROOT / name).read_text(encoding="utf-8", errors="ignore")
            if any(re.search(pattern, text) for pattern in PATTERNS):
                errors.append(f"Potential secret: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Checked {len(list(filter(None, paths)))} tracked paths: no forbidden paths or recognized secrets."
    )


if __name__ == "__main__":
    main()
