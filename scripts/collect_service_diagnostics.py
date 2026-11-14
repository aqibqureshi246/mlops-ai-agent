"""Export bounded, credential-redacted service logs after a live-workflow failure."""

import json
import re
import subprocess
from pathlib import Path

from dotenv import dotenv_values


def redact(text: str, values: dict) -> str:
    for key, value in values.items():
        if (
            value
            and len(value) >= 4
            and any(part in key.upper() for part in ("PASSWORD", "SECRET", "TOKEN", "API_KEY"))
        ):
            text = text.replace(value, "[REDACTED]")
    text = re.sub(r"(?i)Bearer\s+[^\s\"']+", "Bearer [REDACTED]", text)
    return re.sub(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\b", "[REDACTED JWT]", text)


def main():
    values = dotenv_values(".env") if Path(".env").exists() else {}
    observations = []
    for command in (
        ["docker", "compose", "ps", "--all", "--format", "json"],
        ["docker", "compose", "logs", "--no-color", "--tail", "200"],
    ):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=30)
            observations.append(
                {
                    "command": command,
                    "exit_code": result.returncode,
                    "output": redact(result.stdout + result.stderr, values),
                }
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            observations.append({"command": command, "error": type(exc).__name__})
    Path("evidence/service-diagnostics.json").write_text(json.dumps(observations, indent=2) + "\n")
    print("Exported credential-redacted service diagnostics.")


if __name__ == "__main__":
    main()
