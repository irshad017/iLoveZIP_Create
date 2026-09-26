"""
Generate submission.jsonl from canonical 30 test pairs.
magicpin AI Challenge Phase 5 Deliverable.
"""

import json
from pathlib import Path
import bot

def generate():
    dataset_dir = Path("dataset")
    expanded_dir = Path("dataset/expanded")

    # Load categories
    categories = {}
    cat_dir = (expanded_dir / "categories") if (expanded_dir / "categories").exists() else (dataset_dir / "categories")
    for f in cat_dir.glob("*.json"):
        c = json.load(open(f, encoding="utf-8"))
        categories[c.get("slug", f.stem)] = c

    # Load merchants
    merchants = {}
    for f in (expanded_dir / "merchants").glob("*.json"):
        m = json.load(open(f, encoding="utf-8"))
        merchants[m["merchant_id"]] = m

    # Load customers
    customers = {}
    for f in (expanded_dir / "customers").glob("*.json"):
        c = json.load(open(f, encoding="utf-8"))
        customers[c["customer_id"]] = c

    # Load triggers
    triggers = {}
    for f in (expanded_dir / "triggers").glob("*.json"):
        t = json.load(open(f, encoding="utf-8"))
        triggers[t["id"]] = t

    # Load test pairs
    pairs_file = expanded_dir / "test_pairs.json"
    test_pairs = json.load(open(pairs_file, encoding="utf-8"))["pairs"]

    submission_lines = []
    for p in test_pairs:
        tid = p["test_id"]
        trg_id = p["trigger_id"]
        mid = p["merchant_id"]
        cid = p.get("customer_id")

        t = triggers.get(trg_id, {})
        m = merchants.get(mid, {})
        c = customers.get(cid) if cid else None
        cat = categories.get(m.get("category_slug", "retail"), {})

        composed = bot.compose(cat, m, t, c)
        entry = {
            "test_id": tid,
            "body": composed.get("body", ""),
            "cta": composed.get("cta", "binary_yes_no"),
            "send_as": composed.get("send_as", "vera"),
            "suppression_key": composed.get("suppression_key", f"{mid}:{t.get('kind', 'alert')}"),
            "rationale": composed.get("rationale", "")
        }
        submission_lines.append(entry)

    out_path = Path("submission.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for line in submission_lines:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Successfully generated {out_path} with {len(submission_lines)} lines.")

if __name__ == "__main__":
    generate()
