from types import SimpleNamespace

from agent.schemas import ToolCall


class ScriptedPlanner:
    def __init__(self, *batches):
        self.batches = iter(batches)

    def next_calls(self, messages, definitions):
        batch = next(self.batches, [])
        calls = [
            ToolCall(id=f"c{i}", name=name, arguments=args) for i, (name, args) in enumerate(batch)
        ]
        return calls, {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": "{}"}}
                for c in calls
            ],
        }


def make_run(run_id="a" * 32, f1=0.95, status="FINISHED", start=1000, tags=None):
    return SimpleNamespace(
        info=SimpleNamespace(run_id=run_id, status=status, start_time=start, end_time=start + 10),
        data=SimpleNamespace(metrics={"f1_macro": f1}, params={"C": "1.0"}, tags=tags or {}),
    )
