# Offline test fixtures

`v051_best_graph.json` is the unchanged `best_graph` extracted from the local v051 `paired_diagnostics.json` report. It preserves the three-node graph used by `test_fixed_validation.py` without requiring a complete private run archive in a fresh checkout. Tests execute it with synthetic backends; it contains no API credentials or model responses.
