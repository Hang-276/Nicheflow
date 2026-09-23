# v072 execution record: development stage

September 23, 2026. The user authorized starting the experiment and explicitly required progress to survive API quota exhaustion.

## Scope of this launch

This launch implements the first gate of the [v072 protocol](MULTIDOMAIN_V072_EXPERIMENT_PLAN_20260923.md): 12 technical generation checks followed by 960 development answers. Each of four models answers 40 tasks in each of three domains twice. This is **243 planned calls per model**, including its three technical checks.

The process stops after development collection and provisional scoring. Screening, calibration, final evaluation, and the five search rounds do not start automatically. The existing pilot needs a development-result review and all-model mathematical scoring audit before search uses those labels. This is an isolated new runner; historical NicheFlow runs and the serving process are preserved.

Models are L = pinned local Qwen3.5-9B AWQ, M = `qwen3.7-flash-2026-07-15`, H = `qwen3.8-max-0902`, and D = DeepSeek Flash. All use non-thinking requests, an 8,192-token output ceiling, and the v071 provider decoding settings. A shared concise-completion instruction is added for this new protocol. Its effect is not claimed as a controlled stopping-policy improvement.

## Data and scoring

The frozen manifest contains 600 unique tasks: 40 development, 20 screening, 40 calibration, and 100 final tasks per domain. Sampling uses seed 20260923072, source split boundaries, prior-task exclusions, and no model outcomes. MATH is balanced over subject/level where possible; HotpotQA over type/level. Two MATH source records with missing boxed reference answers are excluded before sampling and listed in the manifest.

MBPP uses the sanitized source split intersected with EvalPlus v0.2.0 IDs. There are 107 eligible learning tasks and 224 eligible test tasks after prior-task exclusions. The first source test is public; remaining source tests and EvalPlus base/enhanced tests are private. Source setup imports are preserved. A preparation-only correction added those imports before any generation; it did not alter selected task IDs.

- Mathematics: pinned Math-Verify 0.9.0, latex2sympy2_extended 1.11.0, and SymPy 1.14.0. Nonliteral cases remain provisional and enter an audit queue. Truncations score zero.
- Code: primary quality requires passing the private source tests and EvalPlus base plus enhanced tests. All 40 development reference solutions pass the selected protocol before model inference. Code runs inside a credential-free Linux chroot as UID/GID 65534, with no-new-privileges, a syscall filter blocking network and process-inspection/escape interfaces, a read-only runtime, and CPU/memory/process limits. Only null/zero/random devices and static memory-limit metadata are exposed; host procfs is not mounted. This versioned backend supplements the historical Docker-only backend.
- HotpotQA: official answer F1 is the primary within-domain measure; answer EM, supporting-fact and joint metrics are retained. Gold-answer and supporting-fact checks pass on all development records.

Before launch, every development prompt is tokenized by the existing local service without generating text. The largest prompt is 3,155 local tokens; adding the output allowance fits the 24,576-token service context. Private labels and tests are absent from model-visible requests.

## Persistence and recovery

`nicheflow/v072/durable.py` uses SQLite WAL with `synchronous=FULL`, a single-process writer lock, per-call reservations, immutable request identities, and transactional receipts. Responses are saved **before** local parsing/scoring or aggregate updates. Every 20 completed answers and at stage boundaries the runner creates a consistent SQLite backup.

On a provider refusal or a budget limit, new dispatch pauses. Already dispatched requests are allowed to finish and save their receipts. Completed requests are skipped on explicit resume; rejected requests can be attempted again after the cause is resolved. Timeouts and interrupted requests with unknown outcomes retain their reservations and require reconciliation, not automatic replay. Rejected responses without usage are retained as provider rejections; reference charges are not invoice verification.

The launch has its own conservative caps: **270 attempts per model, CNY 150 and USD 5 in native-currency reference charges**. Max retains the full plan's input/output token ceilings of 20M/10M, but this phase is additionally constrained by its call and currency limits. These caps are not expected spending or purchased credit. Concurrency is four API calls and one local call; scoring is limited to two workers. Each process session has a 12-hour limit and saves progress before stopping.

An end-to-end fake-provider test exhausts quota during the full development matrix, resumes, and confirms 972 completed calls with no repeated successful call, plus the single rejected attempt. Other tests cover in-flight reservations, interrupted calls, changed inputs/protocols, and scoring failures after paid receipts.

## Server operation

Server directory:

```text
/root/autodl-tmp/NicheFlow_Server_Handoff/NicheFlow_Multidomain_v072
```

A separate `.venv` and code-isolation runtime are used. `requirements.lock`, the source hashes, data manifest, local-model identity, and offline readiness evidence are frozen with the run. API credentials remain in the existing private credentials file and are not copied into the runtime tree or Git.

First launch, inside a persistent terminal session:

```bash
bash scripts/launch_v072.sh
```

After inspecting and resolving a recorded pause, explicitly resume the same run:

```bash
bash scripts/launch_v072.sh --resume
```

Do not change the frozen code/configuration or delete the run directory to get past a pause. An unknown in-flight request requires a separate documented reconciliation. Do not restart the local serving process while requests are active.

Run artifacts live in `runs/multidomain_v072/`: `state.sqlite3` and its WAL, `checkpoint.sqlite3`, `progress.json`, `console.log`, `exit.json`, `summary.json`, and `audit_queue.json`. The database and backup are the recovery source; `progress.json` is a display summary. Backups and raw responses remain outside Git.

Offline checks run before paid requests: 167 regression tests, scoring regression cases, code isolation checks, 40 MBPP reference programs, all HotpotQA development references, local model-file hashes, and all 120 development prompt lengths. Test counts and live launch evidence should be recorded with the actual run, not inferred from this document.
