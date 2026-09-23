"""Ten source benchmarks, explicit split roles and model-visible input contracts."""
from __future__ import annotations
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from .spec import EnvironmentBlocked, IntegrityError, digest, file_hash

REGISTRY = {
    "humaneval": {"repo": "openai/openai_humaneval", "config": "openai_humaneval", "split": "test", "role": "evaluation", "evaluator": "python_tests", "capabilities": ["code_sandbox"], "source": "https://github.com/openai/human-eval"},
    "mbpp": {"repo": "google-research-datasets/mbpp", "config": "full", "split": "train", "role": "development", "evaluator": "python_tests", "capabilities": ["code_sandbox"], "source": "https://github.com/google-research/google-research/tree/master/mbpp"},
    "gsm8k": {"repo": "openai/gsm8k", "config": "main", "split": "train", "role": "development", "evaluator": "gsm8k", "capabilities": [], "source": "https://github.com/openai/grade-school-math"},
    "math": {"repo": "EleutherAI/hendrycks_math", "config": "algebra", "split": "train", "role": "development", "evaluator": "math_official", "capabilities": [], "source": "https://github.com/hendrycks/math"},
    "hotpotqa": {"repo": "hotpotqa/hotpot_qa", "config": "distractor", "split": "train", "role": "development", "evaluator": "hotpot_official", "capabilities": [], "source": "https://github.com/hotpotqa/hotpot"},
    "drop": {"repo": "ucinlp/drop", "config": "default", "split": "train", "role": "development", "evaluator": "drop_official", "capabilities": ["scipy"], "source": "https://github.com/allenai/allennlp-models/blob/main/allennlp_models/rc/tools/drop.py"},
    "mmlu_pro": {"repo": "TIGER-Lab/MMLU-Pro", "config": "default", "split": "validation", "role": "development", "evaluator": "choice", "capabilities": [], "source": "https://github.com/TIGER-AI-Lab/MMLU-Pro"},
    "gpqa": {"repo": "Idavidrein/gpqa", "config": "gpqa_main", "split": "train", "role": "evaluation", "evaluator": "choice", "capabilities": ["gated_access"], "source": "https://github.com/idavidrein/gpqa"},
    "gaia": {"repo": "gaia-benchmark/GAIA", "config": "2023_all", "split": "validation", "role": "development", "evaluator": "gaia_official", "capabilities": ["gated_access", "attachments", "tools"], "source": "https://huggingface.co/datasets/gaia-benchmark/GAIA"},
    "travelplanner": {"repo": "osunlp/TravelPlanner", "config": "train", "split": "train", "role": "development", "evaluator": "travelplanner_official", "capabilities": ["travel_database", "official_evaluator"], "source": "https://github.com/OSU-NLP-Group/TravelPlanner"},
}


@dataclass(frozen=True)
class Task:
    id: str
    dataset: str
    split: str
    role: str
    question: str
    gold: object
    evaluator: str
    context: object = None
    options: tuple[str, ...] = ()
    public_tests: tuple[str, ...] = ()
    attachments: tuple[str, ...] = ()
    private: dict | None = None

    def model_input(self):
        return {"question": self.question, "context": self.context, "options": self.options,
                "public_tests": self.public_tests, "attachments": self.attachments,
                "output_contract": {"rational": "Final scalar answer in \\boxed{...}.",
                    "math_official": "Final answer in \\boxed{...}.", "gsm8k": "Final numeric answer in \\boxed{...}.",
                    "choice": "Final choice as ANSWER: A (one letter).",
                    "python_tests": "Return only Python code in a fenced python block.",
                    "hotpot_official": 'Return JSON {"answer": "...", "sp": [["title", 0]]}.',
                    "drop_official": 'Return JSON {"answers": ["span", "span"]}.',
                    "gaia_official": "Return the final short answer only.",
                    "travelplanner_official": "Return a travel plan in the configured official protocol."}[self.evaluator]}

    def require_learning(self):
        if self.role not in {"development", "calibration"}:
            raise IntegrityError(f"{self.id}: held-out data cannot feed adaptive updates")

    def record(self):
        return asdict(self)


def normalize(name, row, index, *, split=None, role=None, seed=20260920):
    spec = REGISTRY[name]
    split, role = split or spec["split"], role or spec["role"]
    if split == "test" and role != "evaluation":
        raise IntegrityError("official test split is evaluation-only")
    if name in {"humaneval", "gpqa"} and role != "evaluation":
        raise IntegrityError("evaluation benchmark cannot silently become training data")
    context, options, public, attachments, private = None, (), (), (), {}
    key = row.get("task_id", row.get("id", row.get("_id", row.get("question_id", index))))
    if name == "humaneval":
        question, gold = row["prompt"], row["canonical_solution"]
        private = {"tests": [row["test"], f"check({row['entry_point']})"], "entry_point": row["entry_point"], "prompt": row["prompt"]}
    elif name == "mbpp":
        question, gold = row["text"], row["code"]
        private = {"tests": list(row["test_list"]), "setup": row.get("test_setup_code", ""), "challenge_tests": row.get("challenge_test_list", [])}
        public = tuple(row["test_list"][:1])  # frozen smoke prompt choice, not an official leaderboard protocol
    elif name == "gsm8k":
        question, gold = row["question"], row["answer"].rsplit("####", 1)[-1].strip()
        private = {"solution": row["answer"]}
    elif name == "math":
        from nicheflow_probe.evaluation import extract_boxed
        question, gold = row["problem"], extract_boxed(row["solution"])
        if gold is None:
            raise ValueError("MATH reference missing boxed answer")
        if "[asy]" in question:
            attachments = ("embedded_asy_requires_renderer",)
        private = {"solution": row["solution"], "level": row.get("level")}
    elif name == "hotpotqa":
        question, gold, context = row["question"], row["answer"], row["context"]
        sp = row["supporting_facts"]
        private = {"supporting_facts": list(zip(sp["title"], sp["sent_id"])) if isinstance(sp, dict) else sp}
    elif name == "drop":
        question, context = row["question"], row["passage"]
        gold = row["answers_spans"]["spans"]
        private = {"answer_types": row["answers_spans"].get("types", [])}
    elif name == "mmlu_pro":
        question, options, gold = row["question"], tuple(row["options"]), row["answer"]
    elif name == "gpqa":
        question = row["Question"]
        values = [(row["Correct Answer"], True)] + [(row[f"Incorrect Answer {i}"], False) for i in range(1, 4)]
        random.Random(int(digest([seed, key])[:16], 16)).shuffle(values)
        options = tuple(v[0] for v in values)
        gold = chr(65 + next(i for i, v in enumerate(values) if v[1]))
    elif name == "gaia":
        question, gold = row["Question"], row.get("Final answer")
        attachments = (row.get("file_path") or row["file_name"],) if row.get("file_name") else ()
        private = {"level": row.get("Level")}
    else:
        question, gold = row["query"], None
        context = row.get("reference_information")
        private = {"official_row": row}
    return Task(f"{name}/{split}/{key}", name, split, role, question, gold, spec["evaluator"],
                context, options, public, attachments, private)


def load_tasks(path):
    tasks = []
    for line in Path(path).read_text().splitlines():
        row = json.loads(line)
        if "dataset" not in row:
            task = Task(row["id"], "math", "train", "development", row["question"], row["gold"], "rational")
        else:
            for k in ("options", "public_tests", "attachments"):
                row[k] = tuple(row.get(k, ()))
            task = Task(**row)
        tasks.append(task)
    if len({t.id for t in tasks}) != len(tasks):
        raise IntegrityError("duplicate task ids")
    return tasks


def fetch_hf(name, directory, *, count, revision, config=None, split=None, seed=20260920):
    """Explicitly invoked only; no auto-download from status or model execution."""
    if not revision or len(revision) != 40:
        raise ValueError("pin the dataset to a full repository commit SHA")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise EnvironmentBlocked("install the data extra in an isolated environment") from exc
    spec = REGISTRY[name]
    directory = Path(directory)
    if directory.exists():
        raise IntegrityError("refusing to overwrite prepared dataset")
    split = split or spec["split"]
    config = config or spec["config"]
    rows = load_dataset(spec["repo"], config, split=split, revision=revision)
    if not 0 < count <= len(rows):
        raise ValueError("invalid sample count")
    indices = sorted(random.Random(seed).sample(range(len(rows)), count))
    raw = [dict(rows[i]) for i in indices]
    tasks = [normalize(name, row, index, split=split, role="evaluation" if split == "test" else None)
             for row, index in zip(raw, indices)]
    directory.mkdir(parents=True)
    for filename, records in [("raw.jsonl", raw), ("tasks.jsonl", [t.record() for t in tasks])]:
        (directory / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    manifest = {"dataset": name, "repo": spec["repo"], "revision": revision,
                "config": config, "split": split, "seed": seed, "indices": indices,
                "task_ids": [t.id for t in tasks], "tasks_sha256": file_hash(directory / "tasks.jsonl"),
                "raw_sha256": file_hash(directory / "raw.jsonl"), "state": "loaded_not_real_smoked"}
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest
