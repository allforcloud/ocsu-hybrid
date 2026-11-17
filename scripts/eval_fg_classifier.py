#!/usr/bin/env python3
"""Evaluate the ConvNeXt-T functional-group head on the validation or test split."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsu_hybrid import paths  # noqa: E402
from ocsu_hybrid.fg.classifier import FGClassifier  # noqa: E402
from ocsu_hybrid.fg.labels import mean_f1, parse_fg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(paths.fg_ckpt()))
    ap.add_argument("--data-json", default=str(paths.data_dir() / "test.json"))
    ap.add_argument("--images-root", default=str(paths.data_dir()))
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="results/fg_test.jsonl")
    args = ap.parse_args()

    records = [r for r in json.loads(Path(args.data_json).read_text(encoding="utf-8"))
               if r["task_name"] == "struct_cap"]
    if args.split == "val":
        random.seed(args.seed)
        random.shuffle(records)
        records = records[:args.val_size]
    if args.limit:
        records = records[:args.limit]

    model = FGClassifier(args.checkpoint, device=args.device)
    images = [Path(args.images_root) / r["images"][0] for r in records]
    preds = model.predict(images, batch_size=args.batch)

    gts = [parse_fg(r["messages"][1]["content"].strip()) for r in records]
    metrics = mean_f1(gts, preds)
    print(json.dumps(dict(metrics, threshold=model.threshold), indent=2))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(json.dumps(dict(
        image=r["images"][0], gt=gt, pred=pred), ensure_ascii=False)
        for r, gt, pred in zip(records, gts, preds)), encoding="utf-8")


if __name__ == "__main__":
    main()
