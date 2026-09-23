"""Fetch a small stratified train-only MATH probe; preserve provenance and raw rows."""
import hashlib
import json
import random
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nicheflow_probe.evaluation import extract_boxed, rational


def fetch(url):
    req = urllib.request.Request(url, headers={"User-Agent": "NicheFlow-prerequisite-probe/0.1"})
    with urllib.request.urlopen(req, timeout=90) as response:
        return json.load(response)


def main():
    config = json.loads((ROOT / "configs/probe.json").read_text())
    raw_dir = ROOT / "data/raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    info = fetch("https://huggingface.co/api/datasets/" + config["dataset"])
    strata = []
    urls = []
    for subject in config["subjects"]:
        rows = []
        for offset in (0, 100, 200, 300):
            query = urllib.parse.urlencode({"dataset": config["dataset"], "config": subject,
                    "split": "train", "offset": offset, "length": 100})
            url = "https://datasets-server.huggingface.co/rows?" + query
            payload = fetch(url)
            (raw_dir / f"{subject}_{offset}.json").write_text(json.dumps(payload, ensure_ascii=False))
            urls.append(url)
            rows.extend(payload["rows"])
            eligible = []
            for level in config["levels"]:
                pool = []
                for row in rows:
                    item = row["row"]
                    gold = extract_boxed(item["solution"])
                    if (item["level"] == level and "[asy]" not in item["problem"]
                            and gold is not None and rational(gold) is not None):
                        pool.append({"id": f"{subject}/train/{row['row_idx']}",
                                     "subject": subject, "level": level,
                                     "question": item["problem"], "gold": gold,
                                     "reference_solution": item["solution"]})
                eligible.append(pool)
            if all(len(pool) >= config["per_stratum"] for pool in eligible):
                break
        if not all(len(pool) >= config["per_stratum"] for pool in eligible):
            raise RuntimeError(f"Insufficient eligible rows for {subject}; no fallback sampling")
        for index, pool in enumerate(eligible):
            random.Random(f"{config['sample_seed']}:{subject}:{index}").shuffle(pool)
            strata.append(pool[:config["per_stratum"]])
    # Wave one covers every subject-level combination; wave two adds its paired sample.
    tasks = [stratum[k] for k in range(config["per_stratum"]) for stratum in strata]
    repeat_ids = random.Random(config["sample_seed"] + 1).sample([t["id"] for t in tasks], config["repeat_count"])
    encoded = "".join(json.dumps(t, ensure_ascii=False) + "\n" for t in tasks)
    (ROOT / "data/tasks.jsonl").write_text(encoded)
    manifest = {"dataset": config["dataset"], "source_revision_at_download": info.get("sha"),
                "source_access": "dataset viewer rows API; exact fetched records preserved and hashed",
                "downloaded_at": datetime.now(timezone.utc).isoformat(), "split": "train",
                "urls": urls, "task_ids": [t["id"] for t in tasks], "repeat_ids": repeat_ids,
                "tasks_sha256": hashlib.sha256(encoded.encode()).hexdigest(), "config": config}
    (ROOT / "data/manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps({"tasks": len(tasks), "repeat_ids": repeat_ids, "sha256": manifest["tasks_sha256"]}))


if __name__ == "__main__":
    main()

