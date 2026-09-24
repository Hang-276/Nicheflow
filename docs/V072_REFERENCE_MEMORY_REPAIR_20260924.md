# v072 scoring infrastructure repair

September 24, 2026. The user authorized repair and completion of the experiment after its final-reference check stopped on `mbpp/test/255`. At the pause there were 5,498 successful model calls and no unknown or in-flight requests. Search, screening and calibration were complete. Deployment candidates and router coefficients were frozen; no final model answers existed.

## Failure and scope

The task's EvalPlus enhanced inputs include 1,663,740 length-77 tuples in one expected output. The retained reference object graph for the enhanced group occupies approximately 1.321 GB. The original 2 GiB address-space cap covered the trusted oracle, its outputs, the scoring runtime and the submitted program. Importing the MBPP deserializer also caused Arrow's default allocator to reserve about 1 GiB of largely unused virtual address space. The oracle therefore failed before model correctness could be evaluated.

After separating reference overhead, a second fault became visible: EvalPlus assigns the next result inside its timed block while still retaining the previous result. Destruction of a very large previous result could consume the next input's 0.2-second minimum allowance. Clearing that previous output before the next timed call made the unchanged canonical solution pass all tests in a targeted diagnostic.

The repair retains every task, test, reference answer, model, prompt, generation limit, chosen workflow and router coefficient. It changes scoring infrastructure through an explicit, recorded migration. It does not silently turn an infrastructure error into a wrong model answer.

## Bounded recovery

- Ordinary scoring retains the original 2 GiB cap, allocator and EvalPlus execution path.
- Only `MemoryError` during trusted reference generation triggers one fresh scoring attempt. Candidate failures, provider failures and unknown API outcomes do not trigger this path.
- The recovery worker has an 8 GiB preparation ceiling and uses Arrow's system allocator. Its EvalPlus child receives a 2 GiB base allowance plus the measured retained reference-object size. Thus its **total process cap is larger than 2 GiB** for reference-heavy tasks; this is an explicit resource-accounting amendment, applied identically to all evaluated models on those tasks.
- In that recovery path only, the previous candidate output is disposed of before the next timed call. The minimum per-call time, reference-time multiplier and overall sandbox deadlines remain bounded. Comparison logic and special oracles are reused from the pinned EvalPlus implementation.
- Network isolation, non-root UID, read-only runtime, seccomp restrictions, CPU/file/process limits and credential exclusion remain in force.
- Further reference OOMs stop with an infrastructure error. No automatic loop or increasing sequence of memory limits is used.

## Validation and provenance

The bounded validation consists of seven focused unit tests, the sandbox isolation self-test, the original failing canonical program and an intentionally wrong program, plus four saved ordinary-path answers with both correct and incorrect outcomes. The clean repaired files passed canonical quality 1 and wrong-program quality 0; ordinary saved scores were unchanged. Earlier diagnostic attempts are not acceptance evidence.

`scripts/migrate_v072_reference_memory.py` verifies acceptance evidence, locks the paused writer, saves the old database and sources, records old/new source and sandbox hashes, then updates only the continuation's frozen infrastructure identity. Phase A's database remains immutable. The continuation accepts only the two named scorer source changes listed in the validated migration; unrelated source changes still fail identity checks.

Previously completed model receipts and scores remain unchanged. The ordinary scoring path does not enter the new recovery branch. Sixty-six successful pre-repair reference checks can be reused because they completed that unchanged path; the repaired task is admitted only with its new passing check. Each remaining final reference check is then journaled separately, so another interruption does not repeat all earlier checks.

Reference identities support the non-finite floats legitimately present in EvalPlus inputs by hashing their stable JSON representation inside the strict journal hash. The first migration preparation encountered this serialization edge case; its database transaction rolled back, its prepared files were restored, and the preparation record was retained before the corrected migration was applied. No model receipt or score was changed by that preparation attempt.

The runtime migration and acceptance records, rather than this design note alone, establish whether the repair has actually been deployed. They are stored at `setup/v072/scoring_memory_migration_20260924.json` and under the continuation run's migration backup.

## Analysis boundary

The final analysis additionally replays the three-single-model pool with only the fixed workflow and with only the searched workflow. This uses the existing frozen predictions and stored final response matrix; it adds no API calls and does not alter deployed choices or select new candidates after seeing final results.
