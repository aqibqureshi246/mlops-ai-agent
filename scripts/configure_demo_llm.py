"""Configure a hosted tool-calling model or the local demonstration model."""

import os
import subprocess
from pathlib import Path

from dotenv import set_key


def main():
    env = Path(".env")
    if os.environ.get("HOSTED_LLM_KEY"):
        url, model = os.environ.get("HOSTED_LLM_URL"), os.environ.get("HOSTED_LLM_MODEL")
        if not url or not model:
            raise SystemExit(
                "Set LLM_BASE_URL and LLM_MODEL repository variables for the hosted model"
            )
        set_key(env, "MLOPS_LLM_API_KEY", os.environ["HOSTED_LLM_KEY"])
        set_key(env, "MLOPS_LLM_BASE_URL", url)
        set_key(env, "MLOPS_LLM_MODEL", model)
        print("Hosted tool-calling model configured without printing credentials.")
    else:
        subprocess.run(
            ["docker", "compose", "--profile", "local-llm", "up", "-d", "ollama"], check=True
        )
        subprocess.run(
            ["docker", "compose", "exec", "-T", "ollama", "ollama", "pull", "qwen2.5:3b"],
            check=True,
        )
        print("Local tool-calling model ready.")


if __name__ == "__main__":
    main()
