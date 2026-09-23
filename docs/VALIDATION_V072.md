# v072: validation questions and launch prerequisites

Prepared on September 23, 2026. This document makes the existing multi-domain plan reviewable before execution. No v072 model requests have been made, and this document is not a runnable experiment configuration. The model decision is fixed: local Qwen3.5-9B AWQ, Qwen3.7 Flash, and Qwen3.8 Max, with DeepSeek Flash as an external baseline and the fixed workflow proposer.

## Questions, comparisons, and interpretation

| Question | Planned comparison | Evidence and interpretation |
|---|---|---|
| Q1. Was the small Max–Flash gap specific to the sampled mathematics tasks? | L, M, H, and DeepSeek on the same unseen tasks in MATH, MBPP, and HotpotQA, with two independent responses per task | Report each domain separately, paired quality differences and uncertainty, disagreement cases, usage, and latency. Compare the pattern of differences across domains; a win in one domain does not establish a universal capability hierarchy. |
| Q2. How much do completion behavior and scoring affect the apparent model gap? | Before the main comparison, check the output contracts, serving template, stop reasons, and versioned scorers on development examples | Record complete-but-wrong, truncated, malformed, infrastructure-failed, and unknown-billing outcomes separately. Audit mathematical equivalence consistently across models. A small technical check can expose a defect, but does not prove a stopping change improves accuracy. |
| Q3. Does collaboration add value over choosing one model? | Three fixed mixed-model seeds against the individual models; evaluate the independently selected fixed workflow and searched workflow on the final set | Report quality–cost trade-offs, including all upstream calls, and compare against DeepSeek. Review correct-to-wrong and wrong-to-correct answer changes where observable. Descriptive transitions are not a controlled causal experiment on reviewer roles. |
| Q4. Do three tiers offer more useful choices than two tiers? | L/H, M/H, and L/M/H single-model candidate pools; then examine the extra value of the retained workflows | Use the same calibration split and routing model class. Report each model's selection frequency, costs, latency, and paired quality differences. Distinguish deployment-model availability from workflow-search effects. A tier's intermediate average accuracy alone is insufficient. |
| Q5. Does short search improve on the initial workflow templates? | Five rounds per domain, at most two new candidates per round; select candidates using independent screening data; compare with fixed seeds on final tasks | Track valid and duplicate proposals, edit types, node counts, model assignments, lineage, and development-to-screening generalization. A searched workflow must offer a useful independent quality–cost trade-off; rising development accuracy alone is insufficient. |
| Q6. Is learned routing more useful than a simple policy? | Always-L/M/H, DeepSeek, a calibration-selected constant choice, two-tier and three-tier routers, and a matched-use random routing reference | Use stored final candidate responses only after routing rules are frozen. Report actual selected-response costs rather than assuming equal usage implies equal cost. Label these results offline replay. A replay cannot establish live cascade latency or throughput. |
| Q7. Is any saving large enough to justify learning the policy? | Separate proposal, search execution, screening, calibration, final evaluation, and deployment costs | Count failed and truncated requests and all upstream tokens. Keep original currencies, freeze any conversion before comparisons, report local GPU time separately, and estimate break-even request volume only when deployment savings are positive. |

The primary empirical comparisons are H versus M within each domain, and retained workflows versus individual models and the fixed workflow. Routing replay is secondary. Report unfavorable and inconclusive results alongside favorable ones; do not remove a task domain because Max does not win.

## Fixed experiment size and sequencing

- Three domains, each with 40 development, 20 screening, 40 calibration, and 100 final tasks: 600 distinct tasks in total.
- Development/screening/calibration use the planned source training splits. Final mathematics excludes MATH-500 and all previously registered questions; MBPP uses its source test split; HotpotQA uses a sealed source development subset.
- Before selection, pin dataset revisions, split assignments, task IDs, hashes, and duplicate checks. MBPP/EvalPlus ID and test coverage must be verified before claiming enhanced-test results; this integration is not currently complete.
- Four individual models, two responses per task. Open each split only at its prescribed stage; do not run the final set early to inspect model differences.
- Three single-model seeds plus three mixed seeds per domain: L draft → M review/final, M draft → H review/final, and H plan → L solve. Reuse compatible single-model receipts rather than paying for them twice.
- Five search rounds per domain, two candidate slots per round, and a fixed 20-task development feedback set. At most three LLM nodes and one Max call per workflow. At most 60 proposer requests overall; invalid/duplicate proposals consume attempts.
- Screen one fixed and two searched candidates per domain. Retain one fixed and one searched workflow, alongside L/M/H. Candidate ranking, tie-breaking, and router settings must be written into the execution configuration before their respective held-out splits are opened.
- Evaluate the final four individual-model baselines and two retained workflows twice per task. Samples are repeated answers to a task, not additional independent tasks.

The first real-model comparison is the development slice: 3 domains × 40 tasks × 4 models × 2 responses = **960 baseline calls**, including **240 Max calls**. At most **30 additional Max calls** are reserved for technical checks using development tasks. Max usage for this opening stage is estimated at about **0.98 million tokens**, already included in the full-plan estimate.

Only proceed to search after the data, model identities, output contracts, scoring, and accounting checks pass. If a protocol changes after development calls, version it and mark incompatible receipts; do not merge different protocols into one result table or silently add reruns. Final evaluation remains sealed.

## Metrics and uncertainty

- **Mathematics:** audited semantic answer accuracy; report truncation and parse failures separately. Preserve the declared text-only Asymptote protocol and report that subset.
- **Code:** proportion of solutions passing the frozen hidden tests, with base/enhanced tests distinguished. Public examples may support reasoning or repair; hidden test cases and gold code must not enter prompts or repair feedback. Sandbox infrastructure failures are not evidence of incorrect code.
- **Multi-hop reading:** official answer F1 as the primary within-domain quality measure; also report answer EM, supporting-fact metrics, and joint metrics. Keep the full required context and sentence identifiers.
- **All domains:** actual input/output tokens, reference API charges, model identity, completion status, mean/P95 call latency, and local GPU time. Reference charges are not verified invoices. Parallel throughput and per-request latency are different measures.
- Compute paired uncertainty by resampling tasks within each domain, carrying both response samples together. Use the two-sample mean per task for comparisons. Show effect sizes and intervals; do not treat overlapping intervals or a nonsignificant result as proof of equivalence.
- A final set of 100 tasks per domain is a pilot, not an adequately powered guarantee of a one-percentage-point non-inferiority margin. Do not combine mathematics accuracy and reading F1 into a single headline score that hides domain differences.

## Readiness inspection of the current repository

These findings come from local source inspection, not a new inspection of the remote server. The existing 157-test suite passed at the preceding Git release; that result covers the historical implementation, not the missing v072 integrations.

| Required preparation | Existing component and remaining work | Required check before launch |
|---|---|---|
| Model adapters and identity | v071 has standalone provider adapters; the main execution path still uses its older profile schema | Confirm the selected three roles and proposer are mapped correctly; freeze non-thinking parameters, node budgets, local weight/runtime identity, and returned API identities |
| Task splits and leakage prevention | Dataset adapters exist; the 600-task v072 manifest does not | Verify source splits, revisions, counts, exclusions, duplicate detection, and that model-visible input omits gold, private solutions, and hidden tests |
| Mathematical scoring | v071's standalone semantic scorer is separate from the main scorer | Integrate a versioned protocol with ordered/unordered answer-type checks and positive/negative regression cases; retain raw outputs for consistent audit |
| Code evaluation | Existing sandbox expects a pinned Docker image; MBPP adapter includes a public example and private tests | Verify isolation on the actual server, public/hidden test separation, ID mapping, correct/reference and deliberately wrong program cases, timeouts, and failure classification |
| Reading evaluation | The current scorer computes several HotpotQA metrics but returns answer EM as a smoke-only quality value | Make the v072 objective explicitly answer F1; test normalization, JSON parsing, sentence indices, and supporting/joint metrics against the pinned evaluator |
| Completion and stopping | v071 used a common 8,192-token ceiling and observed repetition/truncation | Validate serving template, mode and stop behavior on development inputs. Preserve the final-answer contract; do not stop at the first intermediate boxed expression or silently truncate task context |
| Search and deployment candidates | Existing graph/archive machinery uses historical experiment policies | Reinitialize niches for the new model composition; enforce three-node/one-Max limits, record actual mutations and lineage, and keep archive selection distinct from deployment admission |
| Token budget and recovery | Existing journals track requests and costs; the proposed global v072 token limits are not yet integrated | Reserve input/output allowance for concurrent in-flight requests, count proposals and failures, refuse unknown-billing retries, and verify resume does not duplicate completed calls |
| Routing replay | Routing components exist; the new task-domain protocol is not wired | Fit only on calibration data, freeze before final evaluation, forbid gold-based choices, and verify every reported choice has the corresponding stored response |

Required functional and synthetic integration tests must pass before paid execution. Do not interpret this checklist or the translated README as completion of those integrations.

## Budget and claims outside this pilot

The existing ceiling of 4,050 Max calls and planning estimate of **14.44 million input + 6.638 million output tokens** remain unchanged. The proposed hard limits are **20 million input and 10 million output tokens**, with concurrent-call reservations. Other model usage and GPU resources are separate. See the [budget calculation](../reports/multidomain_v072_plan_20260923/max_token_budget.json).

This pilot records niche coverage and lineage, but does **not** isolate the causal benefit of multi-niche search over keeping a single elite. It also does **not** resolve seed dependence: mutation can create new graphs, but that alone does not establish exploration quality. Controlled comparisons of seed sets, parent-free restarts, historical non-elite parents, archive policies, and multiple search seeds remain follow-up experiments with separately specified budgets.

Likewise, adaptive search/calibration scheduling, optimizer substitution, thinking-mode comparisons, and live conditional cascades are outside the current budget. A replay of a cascade is valid only when its requests and visible information match the stored independent responses; a reviewer that consumes earlier answers requires its own execution data.
