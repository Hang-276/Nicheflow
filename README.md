# NicheFlow

**Multi-niche LLM workflow search and routing for quality–cost trade-offs.**

NicheFlow represents workflows as validated graphs, searches over node roles, connections, and model assignments, and preserves candidates with different quality and cost profiles. It investigates how to select a suitable workflow for each request. This repository contains the research prototype, auditable execution and budget accounting, offline tests, historical experiment configurations, and the next experiment protocol.

> **Research status:** v071 model screening on mathematics is complete. The v072 experiment across mathematics, code generation, and multi-hop reading is planned but has not started. Execution adapters and evaluation protocols still require integration. Dataset registration and passing offline tests do not establish real-model performance.

## Project status

| Component or stage | Status | Scope |
|---|---|---|
| Workflow graphs, execution, archives, search, routing, and accounting | Implemented with offline tests | Historical research implementation with documented engineering assumptions |
| v050–v060 | Historical experiments and revisions | MATH workflow experiments, fixed comparisons, recovery, and scoring diagnostics |
| v070–v071 | Independent model screening complete | New model adapters, completion behavior, and audited mathematical answer scoring |
| v072 | **Planned; not running** | Three task domains and five short search rounds per domain |

The main execution path has not yet been migrated to the selected three-model pool and the revised scoring protocols. Running an older configuration does not reproduce v072.

## Model selection

The next experiment uses the following deployment roles. Price and deployment tiers are experimental conditions, not a demonstrated ordering of model capability.

| Role | Configuration |
|---|---|
| L: local model | Qwen3.5-9B with the existing pinned community AWQ weights |
| M: low-cost API | `qwen3.7-flash-2026-07-15` |
| H: high-cost API | `qwen3.8-max-0902` |
| External strong baseline and fixed workflow proposer | DeepSeek Flash, with execution and proposal costs recorded separately |

In v071, each model answered the same 42 fresh, difficult MATH training problems once: L scored **37/42**, M **39/42**, H **41/42**, and DeepSeek **41/42**. All requests used non-thinking mode and an 8,192-token output ceiling. Truncated responses scored zero under the frozen protocol. All five local-model failures and the only Max failure were truncations.

This small sample does not establish that Max consistently outperforms Flash or DeepSeek. DeepSeek remains a required external baseline; savings relative to Max alone are insufficient evidence of overall cost effectiveness. Local inference has no external API charge, while GPU time and rental-cost sensitivity are reported separately.

- [Audited results, costs, and limitations](reports/qwen38_v071_20260923/ANALYSIS_AND_RECOMMENDATIONS.md)
- [Pre-experiment model decision](docs/MODEL_DECISION_20260923.md)
- [Model-selection rationale and related research](docs/THREE_TIER_RESEARCH_SELECTION_V071.md)

Research notes linked from this README are currently in Chinese unless indicated otherwise.

## Offline quick start

Run these commands from the repository root with Python 3.12. The package declares Python 3.10+ support; the release checks below used 3.12. Tests and the fixture require no GPU, model weights, or API credentials and do not make paid inference requests.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .

python -m unittest discover -s tests -v
python -m nicheflow.cli run --mode fixture \
  --config configs/smoke.json --run-dir runs/offline_fixture
python -m nicheflow.cli status runs/offline_fixture
```

The `fixture` mode uses synthetic backends to check execution and accounting. It does not measure model quality or load the historical local-model path in `configs/smoke.json`. Qwen2.5 references in frozen configurations preserve historical experiment identities; Qwen2.5 is no longer part of the deployment plan.

**Packaging verification, September 23, 2026:** all **157 tests passed** in a clean directory exported from the staged repository, using Python 3.12 and NumPy 2.3.5. The fixture and status commands passed, and the authoritative source-document hash matched. These checks did not run real-model or multi-domain experiments.

```bash
python -m nicheflow.cli doctor
python -m nicheflow.cli --help
```

`doctor` reports source-document integrity, optional dependencies, vendored scorers, and code-isolation readiness. Some real-experiment capabilities may be unavailable in an offline installation. Its output does not certify that every experiment protocol has been validated.

Real-model experiments require separately provisioned local serving, API environment variables, pinned model identities, datasets, and scoring dependencies. Read the corresponding experiment protocol first. Historical server scripts contain machine-specific paths and are not a portable one-command deployment interface. Never put API keys in source files, committed configurations, or Git history.

## Repository layout

| Path | Purpose |
|---|---|
| `nicheflow/graph.py`, `runtime.py` | Workflow graphs, contract validation, and execution |
| `nicheflow/archive.py`, `mutations.py`, `search.py` | Archives, candidate mutation, and search components |
| `nicheflow/router.py`, `policy.py`, `research_policy.py` | Routing and versioned research policies |
| `nicheflow/ledger.py`, `main_budget.py` | Event journals, cost accounting, and budget limits |
| `nicheflow/datasets.py`, `scoring.py`, `vendor/` | Dataset adapters and evaluation components |
| `nicheflow_probe/` | Prerequisite and execution-component probes |
| `configs/` | Versioned historical configurations and model-screening plans |
| `scripts/` | Data preparation, evaluation, recovery, and server experiment scripts |
| `tests/` | Offline regression tests and small fixed fixtures |
| `data/` | Dataset snapshots, selections, provenance, and hashes needed for tests and historical reproduction |
| `docs/` | Experiment protocols, engineering records, and validation plans |
| `reports/` | Selected result summaries and budgets, excluding full run journals |

The retained [NicheFlow 3.4 source document](NicheFlow_3.4_技术方案.docx) is checked by the runtime. Engineering additions are recorded in [ENGINEERING_RECORD.md](docs/ENGINEERING_RECORD.md); the implementation is not claimed to be an exact reproduction without additional assumptions.

## Next experiment: v072

The [v072 protocol](docs/MULTIDOMAIN_V072_EXPERIMENT_PLAN_20260923.md) proposes MATH, MBPP, and HotpotQA, with 200 tasks per domain: 40 for development, 20 for screening, 40 for calibration, and 100 for final evaluation. Each domain receives five search rounds with at most two new workflow candidates per round. Final evaluation occurs only after model settings, candidate selection, and routing rules have been frozen.

The [validation and readiness plan](docs/VALIDATION_V072.md) specifies the comparisons, evidence, and launch prerequisites. The main questions are:

1. Does the Max–Flash quality gap vary across mathematics, code, and multi-hop reading?
2. Can reliable completion and scoring distinguish model failures from truncation or evaluation artifacts?
3. Do fixed or searched workflows improve the quality–cost trade-off over individual models, including DeepSeek?
4. Does a three-tier candidate pool offer useful choices beyond a two-tier pool, and can a calibrated router exploit them?
5. Does short workflow search produce candidates that generalize beyond development feedback?

This pilot does not establish multi-seed stability, causal benefits of multi-niche search over single-elite search, or freedom from seed dependence. Those require separate controlled experiments. Routing evaluated from stored candidate responses is explicitly labeled offline replay.

Estimated Max usage is approximately **21 million tokens**: 14.4 million input and 6.6 million output tokens. The proposed allowance is 20 million input plus 10 million output tokens. These are planning estimates, not measured consumption or an already implemented budget guarantee. See the [machine-readable budget](reports/multidomain_v072_plan_20260923/max_token_budget.json).

Remaining implementation work includes completion and stopping behavior, model-adapter integration, semantic math scoring, hidden code tests, complete reading-comprehension metrics, and global token reservations. See the [issue-to-validation matrix](docs/ISSUE_VALIDATION_MATRIX_20260923.md) and [revision plan](docs/NEXT_REVISION_PLAN_20260923.md).

## Reproducibility and artifact boundaries

- Freeze task selections, configurations, model identities, and scoring versions. Gold answers, reference solutions, and hidden tests must never enter model-visible prompts or routing inputs.
- Separate development, screening, calibration, and final evaluation. The v071 selection set is no longer an independent final test set.
- Report domain-specific quality, API input/output usage, GPU time, latency, failures, and truncations. Account for search and calibration separately from deployment.
- Do not automatically retry requests with unknown billing outcomes or resume old runs with changed code. Follow the relevant version's recovery and compatibility protocol.
- Git excludes weights, virtual environments, raw event journals, credentials, old handoff bundles, and personal conversation notes. These remain in local archives; some historical evidence links require those archives.
- `configs/release_manifest.json` and `scripts/verify_release.py` belong to an earlier server handoff. They reference its README and external assets and are not integrity checks for this Git release.

See [data provenance](data/README.md), the [documentation index](docs/README.md), and [third-party notices](THIRD_PARTY_NOTICES.md). Vendored evaluators retain their original licenses. No additional open-source license has been granted for this repository's original code.
