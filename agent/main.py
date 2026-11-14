import argparse
import json
import logging
from uuid import uuid4

from agent.config import Settings
from agent.factory import build_registry
from agent.orchestrator import OpenAIPlanner, Orchestrator


def main():
    parser = argparse.ArgumentParser(description="Evidence-based MLOps agent")
    parser.add_argument("message", nargs="?")
    parser.add_argument(
        "--action", choices=["airflow_trigger_training", "kubernetes_restart_deployment"]
    )
    parser.add_argument("--arguments", default="{}", help="JSON object of exact action parameters")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    settings = Settings()
    registry = build_registry(settings)
    if args.action:
        action = registry.propose(args.action, json.loads(args.arguments), "local-cli")
        print(json.dumps({k: v for k, v in action.items() if k != "confirmation_token"}, indent=2))
        confirmed = input("Type CONFIRM to execute this exact action: ") == "CONFIRM"
        result = registry.confirm(action["confirmation_token"], "local-cli", confirmed, uuid4().hex)
        print(result.model_dump_json(indent=2))
    else:
        if not args.message:
            parser.error("Provide a natural-language request or --action")
        result = Orchestrator(registry, OpenAIPlanner(settings), settings).chat(args.message)
        payload = result.model_dump()
        for action in payload["pending_actions"]:
            action.pop("confirmation_token", None)
        print(json.dumps(payload, indent=2))
        # Tokens are process-local. Give chat users an actionable command for reviewed changes.
        if result.pending_actions:
            print("To review and confirm, run mlops-agent --action <tool> --arguments '<JSON>'.")


if __name__ == "__main__":
    main()
