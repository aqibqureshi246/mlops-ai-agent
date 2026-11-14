"""Export observed pytest JUnit counts for the technical report."""

import json
import platform
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path


def main():
    root = ET.parse(".tmp/test-results.xml").getroot()
    suites = list(root.iter("testsuite"))
    result = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    if result["tests"] < 8 or result["failures"] or result["errors"]:
        raise SystemExit("Passing meaningful test results are required before export")
    result.update(
        {
            "observed_at": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "platform": platform.system(),
            "scope": "Local tests with mocked external services and real temporary sklearn/MLflow checks",
        }
    )
    Path("evidence/automated-tests.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Exported {result['tests']} passing test results.")


if __name__ == "__main__":
    main()
