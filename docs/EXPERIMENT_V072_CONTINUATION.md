# v072 continuation protocol

September 23, 2026. The user requested completion of any remaining experiments before shutting down the rented server. The development run completed 972 requests (12 technical checks and 960 answers). Its 31 nonliteral mathematical labels were reviewed across all four models; no labels changed. The original database remains immutable and is referenced by SHA-256, with a separate audit record.

This continuation implements the remaining bounded pilot: three fixed mixed-model seeds, five search rounds per domain with two candidate slots each, independent screening, calibration, and one frozen final evaluation. It is a separate journal and does not restart the completed development runner.

## Execution and selection

The existing dataset manifest fixes 20 development feedback tasks per domain. Three single-model seeds reuse sample zero from Phase A on exactly those tasks. The mixed seeds are L draft → M review/final, M draft → H review/final, and H plan → L solve. A plan node receives a plan-specific output contract; final solvers see the original task and upstream material as untrusted evidence. Each workflow uses at most three LLM calls and one H call. Complete task context and visible upstream outputs are retained, without silent truncation. A truncated upstream response stops that workflow and counts as a failure with its full cost.

Graphs use a small validated DAG schema: `plan`, `solve`, or `review` roles; L/M/H models; topologically ordered dependencies; no unused nodes. There are no generated execution tools, private-test repair feedback, LLM formatters, or LLM judges. MBPP code is evaluated only in the existing restricted sandbox.

The proposer is DeepSeek Flash, distinct from the execution pool. It sees reusable graph definitions and aggregate development statistics, never reference answers or private tests. Each of 30 candidate slots has at most two proposal attempts. Invalid and duplicate proposals consume attempts. Canonical IDs prevent simple renaming from being counted as novelty. Requested mutation categories are fixed before search: model substitution, prompt editing, deletion/merging, rewiring, and insertion. The requested category is recorded; the resulting graph is the actual intervention and can differ from the request.

Parents are selected by deterministic rotation over occupied niches, described by node count, model set, and chain/branch structure. Each niche supplies its empirical quality elite, with cost and deterministic ID tie-breaks. This bounded pilot does not claim to reproduce every component of the older full NicheFlow adaptive scheduler or establish multi-seed search stability.

Before opening screening data, select one fixed workflow and two searched candidates using development results. Ranking first admits candidates within 0.01 of the best mean quality, then minimizes mean CNY API cost, node count, and ID. Remaining candidates sort by quality. This empirical tolerance is a selection rule, not a non-inferiority guarantee. Screening chooses one of the two searched workflows by the same rule; the fixed workflow was selected on development data. A missing valid search candidate is an explicit failure, not an invented replacement.

Calibration and final evaluation each include four single-model baselines and the retained fixed/searched workflows, twice per task. Three tiers and two workflows form the five-arm deployment pool; DeepSeek is only an external baseline.

## Routing and final-data boundary

Each domain fits one ridge mean-quality predictor and one mean-cost predictor per arm, with regularization 10 and an unpenalized intercept. Four features use only the public question and context: intercept, log question length, digit fraction, and log context length. No private difficulty or subject labels enter routing. The model class and settings are fixed here, without hyperparameter tuning on final data. Later replay will compare constant policies, L/H, M/H, L/M/H, and the five-arm pool. Selection uses the same 0.01 predicted-quality tolerance and lowest predicted CNY cost. It is an offline replay without UCB exploration bonuses, not measured live cascade latency.

Selected graphs, router coefficients, and this protocol are durably committed before final task records are loaded. Final scores cannot change candidates, router coefficients, or model settings. Scoring reference checks do not feed selection. Paired uncertainty resamples tasks, keeping each task's two answers together. Domains are reported separately.

## Audit, budgets, and preservation

Math-Verify and all execution-model settings remain identical to Phase A. New nonliteral mathematical labels are queued for review after seeds, each search round, screening, calibration, and final evaluation. The next phase resumes only with an approval artifact bound to the exact queue digest and a quality decision plus reason for every queued answer. Approved decisions are stored separately and used by subsequent selection; raw scores and paid receipts remain immutable. Previously used decisions cannot silently change. Both candidate proposals in a round use the preceding round's reviewed parent statistics. Pre-audit round snapshots are explicitly labeled; final analysis uses adjudicated labels.

The first seed launch paused on a concurrent SQLite receipt read after two completed calls. A documented source migration serialized that read with the writer lock; both responses were reused. At the first natural audit gate, a second documented migration fixes aggregation order across thread completion, adds explicit per-case adjudication records, and ensures within-round proposals use reviewed prior-round feedback. These changes precede all search proposals and screening/final calls. They do not alter seed prompts, provider settings, or saved seed responses.

The continuation subtracts Phase A use from total ceilings: 4,050 H calls, 20M input and 10M output tokens per model, CNY 150 and USD 5 reference API spend. Other model calls have a separate 15,000-attempt ceiling; proposer attempts remain at most 60. Currency caps may stop the full plan early rather than expanding the budget automatically. Native currency accounting does not claim verified invoices; local GPU rental and inference time are separate.

SQLite durable receipts, per-call worst-case reservations, periodic consistent backups, and explicit resume follow Phase A. Completed requests are reused. Unknown-outcome calls block automatic retry. Audit pauses do not lose completed work. The Phase A and continuation databases together constitute the full record; neither should be deleted to resume.

Server command: `bash scripts/launch_v072_continuation.sh`; resume after resolving a recorded pause with `--resume`. Do not change frozen sources during a run. Final artifacts must be copied to the local workstation and checked before the user shuts down the server.
