#!/usr/bin/env python3
"""Export the Vis-CheBI20 functional-group labels to a reviewable CSV."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsu_hybrid import paths  # noqa: E402
from ocsu_hybrid.fg.labels import parse_fg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-json", default=str(paths.data_dir() / "train.json"))
    ap.add_argument("--out", default="data/fg_labels.csv")
    args = ap.parse_args()

    records = [r for r in json.loads(Path(args.data_json).read_text(encoding="utf-8"))
               if r["task_name"] == "struct_cap"]
    counts = Counter()
    rows = []
    for record in records:
        groups = parse_fg(record["messages"][1]["content"].strip())
        counts.update(groups)
        rows.append((record["images"][0], ";".join(groups)))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image", "functional_groups"])
        writer.writerows(rows)

    print(f"[fg] {len(rows)} samples, {len(counts)} classes -> {out}")
    print("[fg] most frequent:")
    for name, count in counts.most_common(20):
        print(f"  {count:5d}  {name}")


if __name__ == "__main__":
    main()
