"""Parsing and scoring of the OCSU functional-group captions.

The caption is a single sentence, e.g.::

    The functional group ester is highlighted.
    This molecule contains several functional groups, including benzene and carboxylic acid.

``parse_fg`` is the string logic used by the OCSU evaluator; keeping it
byte-compatible matters because the label vocabulary was built from it.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def parse_fg(text: str) -> list[str]:
    """Parse one OCSU functional-group caption into group names."""
    if not text:
        return []
    t = text.strip()
    if t.endswith("."):
        t = t[:-1]
    t = (
        t.replace("The functional group ", "")
        .replace(" is highlighted", "")
        .replace("This molecule contains several functional groups, including ", "")
        .replace(" and", "")
        .replace("This molecule contains the ", "")
        .replace(" functional group", "")
    )
    return [x for x in (part.strip() for part in t.split(", ")) if x]


def mean_f1(gts: Sequence[Iterable[str]], preds: Sequence[Iterable[str]]) -> dict:
    """Mean per-molecule F1, precision, recall and exact-set match.

    The OCSU paper's table swaps the precision/recall columns; F1 is unaffected.
    This function uses the standard definitions.
    """
    f1s, precisions, recalls, exact = [], [], [], []
    for gold, pred in zip(gts, preds):
        g, p = set(gold), set(pred)
        inter = g & p
        precision = len(inter) / len(p) if p else 0.0
        recall = len(inter) / len(g) if g else 0.0
        f1s.append(2 * precision * recall / (precision + recall)
                   if precision + recall else 0.0)
        precisions.append(precision)
        recalls.append(recall)
        exact.append(float(g == p))
    if not f1s:
        return dict(f1=0.0, precision=0.0, recall=0.0, exact=0.0, n=0)
    return dict(
        n=len(f1s),
        f1=float(np.mean(f1s)),
        precision=float(np.mean(precisions)),
        recall=float(np.mean(recalls)),
        exact=float(np.mean(exact)),
    )
