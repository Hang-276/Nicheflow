# Third-party sources

NicheFlow includes third-party evaluation code under its original terms. The original notices are retained next to each implementation:

| Component | Upstream | License file |
|---|---|---|
| MATH equivalence | `hendrycks/math` | [math_equivalence.LICENSE](nicheflow/vendor/math_equivalence.LICENSE) |
| HotpotQA evaluation | `hotpotqa/hotpot` | [hotpot_evaluate.LICENSE](nicheflow/vendor/hotpot_evaluate.LICENSE) |
| DROP evaluation | `allenai/allennlp-models` | [drop_evaluate.LICENSE](nicheflow/vendor/drop_evaluate.LICENSE) |
| GAIA evaluation | `gaia-benchmark/leaderboard` | [gaia_evaluate.LICENSE](nicheflow/vendor/gaia_evaluate.LICENSE) |

Exact upstream revisions, source URLs and hashes are recorded in [vendor/manifest.json](nicheflow/vendor/manifest.json). Data provenance is recorded in [data/README.md](data/README.md) and the accompanying dataset manifests. Model weights are not distributed in this repository; their use is governed by their respective providers' terms.

These notices do not grant a license to NicheFlow's original code.
