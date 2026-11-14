# Evidence exports

This directory contains only genuine exported observations, never illustrative operational results.
Generate five conversations with python scripts/collect_demo.py after all services are healthy.
Generate successful and injected-failure pipeline exports with scripts/run_pipeline.py.
Each record includes timestamps, request identifiers, tool names and returned data.
Mocked responses live only in the test suite and are not demonstration evidence.

Review exported task and pod logs for sensitive content before publication. The agent redacts
configured credentials and sensitive dictionary keys, but third-party log content needs review.
Failure root causes and resolutions belong in the technical report.
