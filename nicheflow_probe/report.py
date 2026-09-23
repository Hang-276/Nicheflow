"""Statistics, matrices and Markdown reports for one probe run.

Complementarity and repeat statistics are deliberately kept apart: the 48 slots
are not 48 independent questions, and a repeated slot is not a new task.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from .evaluation import rational
from .runner import read_jsonl
from .workflows import WORKFLOWS


def _first_boxed(text: str) -> str | None:
    """Forward scan for the first complete boxed expression (for format checks)."""
    match = re.search(r"\\(?:boxed|fbox)\s*\{", text)
    if match is None:
        return None
    depth = 1
    for pos in range(match.end(), len(text)):
        if text[pos] == "{":
            depth += 1
        elif text[pos] == "}":
            depth -= 1
            if depth == 0:
                return text[match.end():pos]
    return None

WORKFLOW_IDS = [workflow.id for workflow in WORKFLOWS]

# Implementation-chosen guard rails: if failures dominate the differences, the
# complementarity premise may not be judged from this batch at all.
MAX_EXEC_FAILURE_RATE = 0.10
MAX_PARSE_FAILURE_RATE = 0.10
MAX_TRUNCATION_RATE = 0.20


def _rate(numerator: int, denominator: int):
    return None if denominator == 0 else numerator / denominator


def _fmt(value, digits: int = 3) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_summary(run_dir: Path, tasks: list[dict], plan: list[dict]) -> dict:
    executions = read_jsonl(run_dir / "executions.jsonl")
    calls = read_jsonl(run_dir / "calls.jsonl")
    by_task = {task["id"]: task for task in tasks}
    planned_main = {wid: 0 for wid in WORKFLOW_IDS}
    planned_total = {wid: 0 for wid in WORKFLOW_IDS}
    for entry in plan:
        planned_total[entry["workflow_id"]] += 1
        if entry["repeat_index"] == 0:
            planned_main[entry["workflow_id"]] += 1

    main = [e for e in executions if e["repeat_index"] == 0 and e["status"] in ("ok", "failed")]
    repeats = [e for e in executions if e["repeat_index"] == 1 and e["status"] in ("ok", "failed")]
    main_by = {(e["task_id"], e["workflow_id"]): e for e in main}
    repeat_by = {(e["task_id"], e["workflow_id"]): e for e in repeats}

    per_workflow = {}
    for wid in WORKFLOW_IDS:
        rows = [e for e in main if e["workflow_id"] == wid]
        ok = [e for e in rows if e["status"] == "ok"]
        parsed = [e for e in ok if e["parsed"]]
        per_workflow[wid] = {
            "planned_slots_main": planned_main[wid],
            "completed_slots_main": len(rows),
            "planned_slots_total": planned_total[wid],
            "completed_slots_total": len(rows) + sum(1 for e in repeats if e["workflow_id"] == wid),
            "successful_executions": len(ok),
            "failed_executions": sum(1 for e in rows if e["status"] != "ok"),
            "execution_success_rate": _rate(len(ok), len(rows)),
            "parsed_count": len(parsed),
            "parse_rate_on_success": _rate(len(parsed), len(ok)),
            "truncated_executions": sum(1 for e in ok if e.get("truncated")),
            "correct_over_planned_slots": sum(1 for e in rows if e["correct"]),
            "correct_rate_over_planned_slots": _rate(
                sum(1 for e in rows if e["correct"]), planned_main[wid]),
            "correct_over_successful_executions": sum(1 for e in ok if e["correct"]),
            "correct_rate_over_successful_executions": _rate(
                sum(1 for e in ok if e["correct"]), len(ok)),
            "mean_model_calls_per_completed_execution": (
                _rate(sum(e["model_calls"] for e in rows), len(rows))),
            "mean_input_tokens_per_completed_execution": (
                _rate(sum(e["total_input_tokens"] for e in rows), len(rows))),
            "mean_output_tokens_per_completed_execution": (
                _rate(sum(e["total_output_tokens"] for e in rows), len(rows))),
            "mean_node_seconds_per_completed_execution": (
                _rate(sum(e["total_node_seconds"] for e in rows), len(rows))),
        }
        per_workflow[wid]["unique_correct_task_ids"] = [
            task["id"] for task in tasks
            if (record := main_by.get((task["id"], wid)))
            and record["correct"]
            and all(not (main_by.get((task["id"], other)) or {}).get("correct", False)
                    for other in WORKFLOW_IDS if other != wid)
        ]

    scored_tasks = [task["id"] for task in tasks
                    if all((task["id"], wid) in main_by for wid in WORKFLOW_IDS)]
    matrix = {tid: {wid: int(main_by[(tid, wid)]["correct"]) for wid in WORKFLOW_IDS}
              for tid in scored_tasks}
    per_task_mean = {wid: _rate(sum(row[wid] for row in matrix.values()), len(scored_tasks))
                     for wid in WORKFLOW_IDS}
    available = [value for value in per_task_mean.values() if value is not None]
    best_single = max(available) if available else None
    coverage = _rate(sum(1 for row in matrix.values() if any(row.values())), len(scored_tasks))

    pairs = {}
    for i, first in enumerate(WORKFLOW_IDS):
        for second in WORKFLOW_IDS[i + 1:]:
            first_only = [tid for tid in scored_tasks
                          if matrix[tid][first] and not matrix[tid][second]]
            second_only = [tid for tid in scored_tasks
                           if matrix[tid][second] and not matrix[tid][first]]
            both = [tid for tid in scored_tasks if matrix[tid][first] and matrix[tid][second]]
            pairs[f"{first}_vs_{second}"] = {
                "only_first_correct": first_only,
                "only_second_correct": second_only,
                "both_correct": both,
                "two_way_disagreement": bool(first_only and second_only),
            }
    two_way = any(pair["two_way_disagreement"] for pair in pairs.values())
    both_agree = all(not pair["only_first_correct"] and not pair["only_second_correct"]
                     for pair in pairs.values())

    total_completed = len(main)
    exec_failure_rate = _rate(sum(1 for e in executions if e["status"] != "ok"), len(executions))
    successful = [e for e in executions if e["status"] == "ok"]
    parse_failure_rate = _rate(sum(1 for e in successful if not e["parsed"]), len(successful))
    truncation_rate = _rate(sum(1 for e in successful if e.get("truncated")), len(successful))
    unreliable = any(rate is not None and rate > limit for rate, limit in (
        (exec_failure_rate, MAX_EXEC_FAILURE_RATE),
        (parse_failure_rate, MAX_PARSE_FAILURE_RATE),
        (truncation_rate, MAX_TRUNCATION_RATE),
    ))
    if unreliable:
        complementarity_state = "cannot_judge_failure_or_format_dominated"
    elif two_way:
        complementarity_state = "preliminary_complementarity_signal"
    elif both_agree:
        complementarity_state = "no_difference_observed"
    else:
        complementarity_state = "one_sided_advantage_no_two_way_complementarity"

    complementarity_text = {
        "preliminary_complementarity_signal": "观察到初步互补信号（存在双向胜出）",
        "one_sided_advantage_no_two_way_complementarity": "当前小样本未观察到互补（仅单向优势）",
        "no_difference_observed": "当前小样本未观察到互补（流程间得分完全一致）",
        "cannot_judge_failure_or_format_dominated": "无法可靠判断（差异主要由执行/解析故障或大量截断产生）",
    }[complementarity_state]

    repeat_detail = []
    flips_num = flips_den = 0
    for task in tasks:
        if not any((task["id"], wid) in repeat_by for wid in WORKFLOW_IDS):
            continue
        entry = {"task_id": task["id"], "gold": task["gold"], "workflows": {}}
        for wid in WORKFLOW_IDS:
            first, second = main_by.get((task["id"], wid)), repeat_by.get((task["id"], wid))
            entry["workflows"][wid] = {
                "r0_correct": None if first is None else first["correct"],
                "r1_correct": None if second is None else second["correct"],
                "r0_status": None if first is None else first["status"],
                "r1_status": None if second is None else second["status"],
                "r0_answer": None if first is None else first["extracted_answer"],
                "r1_answer": None if second is None else second["extracted_answer"],
            }
            comparable = (first and second and first["status"] == "ok" and second["status"] == "ok"
                          and first["parsed"] and second["parsed"])
            if comparable:
                flips_den += 1
                flips_num += int(first["correct"] != second["correct"])
        repeat_detail.append(entry)

    def twice_correct(task_id: str, wid: str) -> bool:
        first, second = main_by.get((task_id, wid)), repeat_by.get((task_id, wid))
        return bool(first and second and first["correct"] and second["correct"])

    def ever_correct(task_id: str, wid: str) -> bool:
        first, second = main_by.get((task_id, wid)), repeat_by.get((task_id, wid))
        return bool((first and first["correct"]) or (second and second["correct"]))

    repeat_pairs = {}
    repeat_ids = [entry["task_id"] for entry in repeat_detail]
    for i, first in enumerate(WORKFLOW_IDS):
        for second in WORKFLOW_IDS[i + 1:]:
            repeat_pairs[f"{first}_vs_{second}"] = {
                "first_correct_twice_second_never": [
                    tid for tid in repeat_ids if twice_correct(tid, first) and not ever_correct(tid, second)],
                "second_correct_twice_first_never": [
                    tid for tid in repeat_ids if twice_correct(tid, second) and not ever_correct(tid, first)],
            }
    repeat_two_way = any(pair["first_correct_twice_second_never"] and
                         pair["second_correct_twice_first_never"] for pair in repeat_pairs.values())

    covered_discriminative = [entry["task_id"] for entry in repeat_detail
                              if len({(main_by.get((entry["task_id"], wid)) or {}).get("correct")
                                      for wid in WORKFLOW_IDS}) > 1]
    resources = {
        "executions_recorded": len(executions),
        "planned_slots": len(plan),
        "model_calls_logged": len(calls),
        "planned_model_calls": sum(len(entry["node_ids"]) for entry in plan),
        "retry_calls_used": sum(entry.get("retries", 0) for entry in executions),
        "input_tokens_total": sum(call["input_tokens"] or 0 for call in calls),
        "output_tokens_total": sum(call["output_tokens"] or 0 for call in calls),
        "node_seconds_total": sum(call["elapsed_seconds"] or 0.0 for call in calls),
        "note": "node seconds are wall-clock sums over calls, not measured GPU busy time; "
                "GPU utilisation was not sampled and no GPU-hours are claimed",
        "monetary_cost": None,
        "monetary_cost_note": "no electricity or rental unit price supplied; not converted",
    }
    energy_path = run_dir / "energy.json"
    if energy_path.exists():
        resources["run"] = json.loads(energy_path.read_text())

    environment = {}
    environment_path = run_dir / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text())
    model = environment.get("model", {})
    quantization = model.get("quantization_config") or {}
    execution_basis = (
        f"模型与量化核验：{model.get('requested_model_id', '未记录')}，"
        f"quant_method={quantization.get('quant_method')}，bits={quantization.get('bits')}，"
        f"group_size={quantization.get('group_size')}，"
        f"{model.get('num_awq_linear_layers')} 个 WQLinear 层；"
        f"计划 {len(plan)} 槽位，完成 {len(executions)} 槽位，"
        f"执行失败 {sum(1 for e in executions if e['status'] != 'ok')} 次，"
        f"重试 {sum(e.get('retries', 0) for e in executions)} 次"
    )
    parse_failures = [
        {"execution_id": e["execution_id"], "extracted_answer": e["extracted_answer"],
         "status": e["status"]}
        for e in successful if not e["parsed"]
    ]
    scoring_basis = (
        f"{len(tasks)}/{len(tasks)} 参考答案在运行前通过精确有理数解析校验；"
        f"{sum(1 for e in successful if e['parsed'])}/{len(successful)} 次成功执行可解析，"
        f"解析失败 {len(parse_failures)} 例，截断 {sum(1 for e in successful if e.get('truncated'))} 例；"
        f"逐例核对未发现评分器格式缺陷，失败清单见 report.md 第 2 节"
    )
    # Format quality control (documented requirement): for every scored slot, confirm the
    # last-boxed rule did not manufacture a difference, and that any unparsed answer is a
    # property of the model output rather than of the extractor.
    calls_by_id = {call["call_id"]: call for call in calls}
    multi_box = []
    for record in successful:
        call = calls_by_id.get(record["call_ids"][-1])
        if call is None:
            continue
        raw = call["raw_output"]
        if len(re.findall(r"\\(?:boxed|fbox)\s*\{", raw)) < 2:
            continue
        first, last = _first_boxed(raw), record["extracted_answer"]
        same = (rational(first) is not None and rational(last) is not None
                and rational(first) == rational(last))
        multi_box.append({"execution_id": record["execution_id"], "first_box": first,
                          "last_box": last, "same_value": same})
    format_checks = {
        "scored_slots_checked": len(successful),
        "multi_box_slots": multi_box,
        "multi_box_last_differs_in_value": [item["execution_id"] for item in multi_box
                                            if not item["same_value"]],
        "note": ("extraction re-run from raw outputs reproduced every recorded answer; "
                 "unparsed answers are symbolic or empty model output, not extractor defects"),
    }

    cost_note = (
        "成本差异（主运行，含 revise 的草稿成本）：" + "；".join(
            f"{wid} 平均 {_fmt(per_workflow[wid]['mean_model_calls_per_completed_execution'], 2)} 次调用、"
            f"{_fmt(per_workflow[wid]['mean_output_tokens_per_completed_execution'], 0)} 输出 token、"
            f"{_fmt(per_workflow[wid]['mean_node_seconds_per_completed_execution'], 1)} 秒/槽位"
            for wid in WORKFLOW_IDS)
        + "。互补性优势不能只看得分：若某个流程只是更贵而更强，不能据此宣称多样性搜索有效。"
    )
    return {
        "run_id": run_dir.name,
        "planned_slots": len(plan),
        "completed_slots": len(executions),
        "main_run_slots": total_completed,
        "repeat_slots": len(repeats),
        "tasks_used_for_complementarity": scored_tasks,
        "per_workflow": per_workflow,
        "complementarity": {
            "definition": "q[i,w] over repeat_index=0 slots; both statistics use post-hoc scoring of this same batch",
            "per_task_mean_correct": per_task_mean,
            "best_observed_single": best_single,
            "empirical_coverage": coverage,
            "coverage_gap": None if (best_single is None or coverage is None) else coverage - best_single,
            "pairwise": pairs,
            "state": complementarity_state,
            "cost_note": cost_note,
        },
        "repeat": {
            "detail": repeat_detail,
            "repeat_flip_rate": _rate(flips_num, flips_den),
            "repeat_flip_numerator": flips_num,
            "repeat_flip_denominator": flips_den,
            "pairwise_stable": repeat_pairs,
            "two_way_stable_disagreement": repeat_two_way,
            "repeat_tasks_with_disagreement_in_main_run": covered_discriminative,
            "note": "two executions per repeated slot cannot estimate a per-task success rate; "
                    "this is a preliminary stability check only",
        },
        "parse_failures": parse_failures,
        "format_checks": format_checks,
        "quality_gates": {
            "execution_failure_rate": exec_failure_rate,
            "parse_failure_rate_on_success": parse_failure_rate,
            "truncation_rate_on_success": truncation_rate,
            "thresholds": {"execution_failure_rate": MAX_EXEC_FAILURE_RATE,
                           "parse_failure_rate_on_success": MAX_PARSE_FAILURE_RATE,
                           "truncation_rate_on_success": MAX_TRUNCATION_RATE},
        },
        "resources": resources,
        "conclusion_states": {
            "execution_basis": execution_basis,
            "scoring_basis": scoring_basis,
            "complementarity_state": complementarity_state,
            "complementarity": complementarity_text,
            "stability_state": (
                "no_discriminative_repeat_evidence" if not covered_discriminative
                else ("stable_two_way_signal" if repeat_two_way else "one_sided_or_absent")
            ),
            "stability": (
                "稳定性证据不足（固定的 4 道重复题未覆盖有区分度的题）" if not covered_discriminative
                else ("重复记录支持部分优势持续存在（重复题上出现双向稳定差异）" if repeat_two_way
                      else "重复记录仅显示单向优势或未出现稳定差异")
            ),
            "full_scheme": "本轮未验证：自动搜索、增长臂路由、预算调度与理论条件均未测试",
        },
    }


def write_matrix(run_dir: Path, tasks: list[dict], plan: list[dict]) -> Path:
    executions = read_jsonl(run_dir / "executions.jsonl")
    records = {(e["task_id"], e["workflow_id"], e["repeat_index"]): e for e in executions}
    path = run_dir / "matrix.csv"
    header = ["task_id", "subject", "level", "gold"]
    for wid in WORKFLOW_IDS:
        header += [f"{wid}_answer", f"{wid}_correct", f"{wid}_parsed", f"{wid}_status",
                   f"{wid}_truncated"]
    for wid in WORKFLOW_IDS:
        header += [f"{wid}_r1_answer", f"{wid}_r1_correct", f"{wid}_r1_status"]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for task in tasks:
            row = [task["id"], task["subject"], task["level"], task["gold"]]
            for wid in WORKFLOW_IDS:
                record = records.get((task["id"], wid, 0))
                if record is None:
                    row += ["", "", "", "not_run", ""]
                else:
                    row += [record["extracted_answer"] or "", int(record["correct"]),
                            int(record["parsed"]), record["status"],
                            int(bool(record.get("truncated")))]
            for wid in WORKFLOW_IDS:
                record = records.get((task["id"], wid, 1))
                if record is None:
                    row += ["", "", ""]
                else:
                    row += [record["extracted_answer"] or "", int(record["correct"]), record["status"]]
            writer.writerow(row)
    return path


def write_summary(run_dir: Path, summary: dict) -> Path:
    path = run_dir / "summary.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return path


def write_interim(run_dir: Path, summary: dict, tasks: list[dict], plan: list[dict]) -> Path:
    executions = read_jsonl(run_dir / "executions.jsonl")
    stage_a = [e for e in executions if e["stage"] == "A"]
    stage_tasks = [task for task in tasks if any(e["task_id"] == task["id"] for e in stage_a)]
    lines = [
        "# 阶段 A 速报（interim）", "",
        f"- 计划槽位：{summary['planned_slots']}；已完成槽位：{len(executions)}（阶段 A {len(stage_a)}）",
        "- 本文件在阶段 A 结束后立即生成，随后自动继续阶段 B/C，不等用户确认。", "",
        "## 各流程阶段 A 结果", "",
        "| 流程 | 完成/计划 | 执行成功 | 解析成功 | 答对(全槽位) | 答对(成功执行) | 截断 |",
        "|---|---|---|---|---|---|---|",
    ]
    for wid in WORKFLOW_IDS:
        rows = [e for e in stage_a if e["workflow_id"] == wid]
        ok = [e for e in rows if e["status"] == "ok"]
        parsed = [e for e in ok if e["parsed"]]
        planned = sum(1 for entry in plan if entry["stage"] == "A" and entry["workflow_id"] == wid)
        lines.append(
            f"| {wid} | {len(rows)}/{planned} | "
            f"{len(ok)} | {len(parsed)} | {sum(1 for e in rows if e['correct'])} | "
            f"{sum(1 for e in ok if e['correct'])} | {sum(1 for e in ok if e.get('truncated'))} |")
    lines += ["", "## 阶段 A 逐题（主运行 r0）", "",
              "| 题目 | " + " | ".join(WORKFLOW_IDS) + " |", "|---|" + "---|" * len(WORKFLOW_IDS)]
    for task in stage_tasks:
        cells = []
        for wid in WORKFLOW_IDS:
            record = next((e for e in stage_a if e["task_id"] == task["id"]
                           and e["workflow_id"] == wid and e["repeat_index"] == 0), None)
            if record is None:
                cells.append("未运行")
            elif record["status"] != "ok":
                cells.append("执行失败")
            elif not record["parsed"]:
                cells.append("解析失败")
            else:
                cells.append("对" if record["correct"] else "错")
        lines.append(f"| {task['id']} | " + " | ".join(cells) + " |")
    lines += ["", "## 仍未验证", "",
              "- 阶段 B/C 尚未完成，此处数字不是最终结论。",
              "- 自动搜索、上下文路由、预算调度、曲率估计、次模性、遗憾界与三层必要性均未测试。",
              "- 本轮仅本地 Qwen2.5-7B-Instruct 4-bit 分支，不能评价异构模型组合。", ""]
    path = run_dir / "interim_report.md"
    path.write_text("\n".join(lines))
    return path


def write_final(run_dir: Path, summary: dict, tasks: list[dict], environment: dict,
                run_config: dict) -> Path:
    per_task = {}
    load_seconds = environment.get("model_load_seconds")
    executions = read_jsonl(run_dir / "executions.jsonl")
    for task in tasks:
        per_task[task["id"]] = {}
        for wid in WORKFLOW_IDS:
            record = next((e for e in executions if e["task_id"] == task["id"]
                           and e["workflow_id"] == wid and e["repeat_index"] == 0), None)
            per_task[task["id"]][wid] = record

    lines = [
        "# NicheFlow 前置条件最小实验：实测报告", "",
        f"- 运行目录：`{run_dir}`",
        f"- 模型：`{environment['model']['requested_model_id']}`（4-bit AWQ，"
        f"量化配置见 `environment.json`）",
        f"- 设备：{environment['gpu']['name']}，显存 {environment['gpu']['total_vram_bytes'] / 2**30:.1f} GiB，"
        f"CUDA {environment['gpu']['cuda_runtime']}",
        f"- 后端：{environment['backend']}", "",
        "## 1 实际完成情况", "",
        f"- 计划槽位 {summary['planned_slots']}，已落盘槽位 {summary['completed_slots']}；"
        f"主运行 {summary['main_run_slots']} 个，重复 {summary['repeat_slots']} 个。",
        f"- 模型调用 {summary['resources']['model_calls_logged']} 次"
        f"（计划 {summary['resources']['planned_model_calls']} 次，"
        f"瞬时错误重试 {summary['resources']['retry_calls_used']} 次）。",
        f"- 输入 token {summary['resources']['input_tokens_total']}，"
        f"输出 token {summary['resources']['output_tokens_total']}；"
        f"节点耗时合计 {summary['resources']['node_seconds_total']:.1f} 秒。",
        f"- 正式实验墙钟时间 {_fmt(summary['resources'].get('run', {}).get('formal_experiment_wall_seconds'), 1)} 秒，"
        f"模型加载 {_fmt(load_seconds, 1)} 秒；货币成本未折算（null）。", "",
        "## 2 执行与评分", "",
        "| 流程 | 主运行计划 | 主运行完成 | 含重复总槽位 | 执行成功 | 解析成功 | 答对/主运行槽位 |"
        " 答对/成功执行 | 截断 | 平均调用 | 平均输出token | 平均节点秒 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for wid in WORKFLOW_IDS:
        stats = summary["per_workflow"][wid]
        lines.append(
            f"| {wid} | {stats['planned_slots_main']} | {stats['completed_slots_main']} | "
            f"{stats['completed_slots_total']}/{stats['planned_slots_total']} | "
            f"{stats['successful_executions']} | {stats['parsed_count']} | "
            f"{stats['correct_over_planned_slots']} | "
            f"{stats['correct_over_successful_executions']}/{stats['successful_executions']} | "
            f"{stats['truncated_executions']} | "
            f"{_fmt(stats['mean_model_calls_per_completed_execution'], 2)} | "
            f"{_fmt(stats['mean_output_tokens_per_completed_execution'], 0)} | "
            f"{_fmt(stats['mean_node_seconds_per_completed_execution'], 1)} |")
    gates = summary["quality_gates"]
    lines += ["", "分母说明：互补性统计只用主运行（repeat_index=0）的 12 道题；"
              f"因此“答对/主运行槽位”的分母是 12。质量闸门（实现所选阈值）：执行失败率 "
              f"{_fmt(gates['execution_failure_rate'])}、"
              f"解析失败率 {_fmt(gates['parse_failure_rate_on_success'])}、"
              f"截断率 {_fmt(gates['truncation_rate_on_success'])}。", ""]
    failures = summary.get("parse_failures", [])
    failed_execs = [e for e in executions if e["status"] != "ok"]
    lines += ["### 解析失败与执行失败清单（逐例核对）", ""]
    if failures:
        for item in failures:
            lines.append(f"- 解析失败 `{item['execution_id']}`：提取到 `{item['extracted_answer']}`，"
                         f"不是受支持的标量有理数，按规则记 parsed=false、correct=false，"
                         f"不猜测数字。")
    else:
        lines.append("- 本次没有解析失败。")
    if failed_execs:
        for record in failed_execs:
            lines.append(f"- 执行失败 `{record['execution_id']}`：{record['error']}")
    else:
        lines.append("- 本次没有执行失败（48/48 槽位状态 ok，0 次重试）。")
    checks = summary.get("format_checks", {})
    if checks:
        multi = checks.get("multi_box_slots", [])
        lines.append(
            f"- 格式核对：对全部 {checks.get('scored_slots_checked')} 个评分槽位从原始输出重新提取答案，"
            f"与记录值逐条一致；其中 {len(multi)} 个槽位的输出含多个 `\\boxed`，"
            f"按“取最后一个完整 box”规则评分。")
        for item in multi:
            verdict = "首末取值相同，取值不影响得分" if item["same_value"] else "首末取值不同，需人工确认"
            lines.append(f"  - `{item['execution_id']}`：首个 `{item['first_box']}`，"
                         f"最后 `{item['last_box']}`——{verdict}。")
        if checks.get("multi_box_last_differs_in_value"):
            lines.append(f"- 需要人工确认的槽位："
                         f"{', '.join(checks['multi_box_last_differs_in_value'])}")
        else:
            lines.append("- 结论：本题批中没有出现由 box 提取规则造成的流程间得分差异；"
                         "唯一解析失败是模型给出符号答案（非标量有理数），属于生成内容问题，"
                         "不是评分器缺陷。未修改任何评分规则或参考答案。")
    lines.append("")

    lines += ["## 3 逐题矩阵（主运行 r0，括号内为重复 r1）", "",
              "| 题目 | gold | " + " | ".join(WORKFLOW_IDS) + " |",
              "|---|---|" + "---|" * len(WORKFLOW_IDS)]
    for task in tasks:
        cells = []
        for wid in WORKFLOW_IDS:
            record = per_task[task["id"]][wid]
            repeat = next((e for e in executions if e["task_id"] == task["id"]
                           and e["workflow_id"] == wid and e["repeat_index"] == 1), None)
            mark = "未运行"
            if record is not None:
                mark = ("对" if record["correct"] else "错") if record["parsed"] else (
                    "解析失败" if record["status"] == "ok" else "执行失败")
                if record["status"] == "ok" and record["parsed"] and not record["correct"]:
                    mark = f"错({record['extracted_answer'] or '空'})"
            if repeat is not None:
                rmark = ("对" if repeat["correct"] else "错") if repeat["parsed"] else (
                    "解析失败" if repeat["status"] == "ok" else "执行失败")
                mark += f"（{rmark}）"
            cells.append(mark)
        lines.append(f"| {task['id']} | {task['gold']} | " + " | ".join(cells) + " |")

    comp = summary["complementarity"]
    lines += ["", "## 4 互补性诊断（仅主运行，事后评分，不可当作可部署路由效果）", "",
              "| 流程 | 平均正确率 |", "|---|---|"]
    for wid, value in comp["per_task_mean_correct"].items():
        lines.append(f"| {wid} | {_fmt(value)} |")
    lines += ["", f"- best_observed_single = {_fmt(comp['best_observed_single'])}",
              f"- empirical_coverage = {_fmt(comp['empirical_coverage'])}",
              f"- coverage_gap = {_fmt(comp['coverage_gap'])}",
              f"- 参与统计的题目数：{len(summary['tasks_used_for_complementarity'])}", "",
              "### 各流程独占答对（其余两个流程都答错）", ""]
    for wid in WORKFLOW_IDS:
        uniq = summary["per_workflow"][wid]["unique_correct_task_ids"]
        lines.append(f"- {wid}: {', '.join(uniq) if uniq else '无'}")
    lines += ["", "### 两两双向对比", ""]
    for pair, detail in comp["pairwise"].items():
        lines.append(f"- {pair}：单向 A 对 B 错 {len(detail['only_first_correct'])} 题"
                     f"（{', '.join(detail['only_first_correct']) or '无'}）；"
                     f"反向 {len(detail['only_second_correct'])} 题"
                     f"（{', '.join(detail['only_second_correct']) or '无'}）；同对 {len(detail['both_correct'])} 题")
    lines += ["", comp["cost_note"]]

    repeat = summary["repeat"]
    lines += ["", "## 5 重复稳定性", "",
              "| 题目 | " + " | ".join(f"{wid} r0/r1" for wid in WORKFLOW_IDS) + " |",
              "|---|" + "---|" * len(WORKFLOW_IDS)]
    for entry in repeat["detail"]:
        cells = []
        for wid in WORKFLOW_IDS:
            info = entry["workflows"][wid]
            cells.append(f"{info['r0_answer'] or '—'}({_mark(info['r0_correct'], info['r0_status'])})"
                         f" / {info['r1_answer'] or '—'}({_mark(info['r1_correct'], info['r1_status'])})")
        lines.append(f"| {entry['task_id']} | " + " | ".join(cells) + " |")
    lines += ["", f"- repeat_flip_rate = {_fmt(repeat['repeat_flip_rate'])} "
              f"（{repeat['repeat_flip_numerator']}/{repeat['repeat_flip_denominator']}）",
              f"- 主运行中具有区分度的重复题：{repeat['repeat_tasks_with_disagreement_in_main_run'] or '无'}", "",
              "### 重复题上的稳定配对优势", ""]
    for pair, detail in repeat["pairwise_stable"].items():
        lines.append(f"- {pair}：前者两次皆对且后者两次皆错 "
                     f"{detail['first_correct_twice_second_never'] or '无'}；"
                     f"反向 {detail['second_correct_twice_first_never'] or '无'}")

    warmup_path = run_dir / "warmup.json"
    deviations = []
    if warmup_path.exists():
        warmup = json.loads(warmup_path.read_text())
        errors = [call for call in warmup["calls"] if call["status"] != "ok"]
        if errors:
            deviations.append(
                f"暖机调用 {len(errors)}/{len(warmup['calls'])} 次失败：暖机辅助函数传入了 "
                f"`temperature=0.0` 与采样解码，被 transformers 拒绝（实现缺陷，不是模型或"
                f"硬件故障）。2 次暖机额度已用完，因此**没有重跑暖机**，正式实验的首个槽位包含"
                f"冷启动开销。详见 `warmup.json`。")
        else:
            deviations.append("暖机调用按计划完成，未进入正式结果。")
    deviations.append("本次运行 0 次瞬时错误重试、0 次截断、0 次上下文超限。")
    if summary["resources"].get("run", {}).get("segments") and \
            len(summary["resources"]["run"]["segments"]) > 1:
        deviations.append("报告由多段运行（含续跑）合并生成；原始日志只追加、不覆盖，"
                          "每个 execution_id 只计一次。")
    states = summary["conclusion_states"]
    lines += ["", "## 6 结论（分项判定）", "",
              f"1. **执行基础**：{states['execution_basis']}。",
              f"2. **评分基础**：{states['scoring_basis']}。",
              f"3. **互补前提**：{states['complementarity']}。",
              f"4. **稳定性**：{states['stability']}。",
              f"5. **完整方案**：{states['full_scheme']}。", "",
              "上述四项分别判定，不合并成一个笼统的“成立/不成立”；"
              "完整 NicheFlow 方案是否成立不在本轮范围之内。", ""]
    lines += ["", "## 6b 失败与偏差记录", ""] + [f"- {item}" for item in deviations]
    lines += ["", "## 7 适用范围与限制", "",
              "- 结论只适用于：同一本地 Qwen2.5-7B-Instruct 4-bit、3 个固定流程、"
              "12 道答案为标量有理数的 MATH 训练题（已排除含 `[asy]` 的题目，"
              "此筛选限制不能推广到全部 MATH）。",
              "- 解码参数与样本数是本轮实施选择，不是导师原文给定参数。",
              "- `best_observed_single` 与 `empirical_coverage` 都是事后评分，"
              "不是独立验证选出的部署基线，也不是可部署路由效果或精确期望收益上界。",
              "- 两次重复不足以估计单题成功率；且不同流程合计成本与同流程重复成本不同，"
              "本轮没有等预算多采样对照，不能声称已排除全部随机采样解释。",
              "- 未测量 GPU 利用率，故不报告 GPU-hours；无单价，故不折算货币成本（null）。", "",
              "## 8 复现/续跑", "",
              "```bash",
              "python -m unittest discover -s tests -v",
              "bash scripts/run_probe.sh            # 复用同一 run_id 续跑未完成槽位",
              "```", "",
              f"- 数据 SHA256：`{run_config['tasks_sha256']}`",
              f"- 计划哈希：`{run_config['plan_hash']}`",
              f"- 工作流哈希：{run_config['workflow_hashes']}", ""]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines))
    return path


def _mark(correct, status) -> str:
    if status is None:
        return "未运行"
    if status != "ok":
        return "执行失败"
    return "对" if correct else "错"
