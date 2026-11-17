"""Functional-group metrics following the OCSU evaluator."""
from __future__ import annotations

from typing import Dict, Iterable

from ..fg.labels import mean_f1


def score_fg(records: Iterable[dict], predictions: Dict[str, list[str]]
             ) -> tuple[dict, list[dict]]:
    rows = []
    for record in records:
        image = record["images"][0]
        gold = record["gt"]
        pred = predictions.get(image) or []
        metrics = mean_f1([gold], [pred])
        rows.append(dict(
            image=image,
            gt=sorted(set(gold)),
            pred=sorted(set(pred)),
            f1=metrics["f1"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            exact=metrics["exact"],
        ))
    if not rows:
        return dict(n=0, f1=0.0, precision=0.0, recall=0.0, exact=0.0), rows
    return dict(
        n=len(rows),
        f1=sum(row["f1"] for row in rows) / len(rows),
        precision=sum(row["precision"] for row in rows) / len(rows),
        recall=sum(row["recall"] for row in rows) / len(rows),
        exact=sum(row["exact"] for row in rows) / len(rows),
    ), rows
