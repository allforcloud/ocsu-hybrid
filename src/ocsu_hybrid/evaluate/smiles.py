"""SMILES metrics following the OCSU ``SmilesEvaluator``.

- ``Acc_s``: canonical exact match with double-bond cis/trans ignored.
- ``Acc_g``: graph exact match with chirality and cis/trans ignored.
- ``Acc_c``: ``Acc_s`` restricted to molecules whose ground truth has chirality.
- ``Tanimoto``: RDKit fingerprint similarity.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
from rdkit import Chem, DataStructs, RDLogger

RDLogger.DisableLog("rdApp.*")


def strip_prefix(text: str, prefix: str) -> str:
    t = (text or "").strip()
    if t.lower().startswith(prefix.lower()):
        t = t[len(prefix):]
    return t.strip().rstrip(".").strip()


def canonicalize(smiles: Any, ignore_chiral: bool = False,
                 ignore_cistrans: bool = False,
                 replace_rgroup: bool = True) -> tuple[str, bool]:
    """Return ``(canonical_smiles, success)`` exactly like the OCSU evaluator."""
    if not isinstance(smiles, str) or not smiles:
        return "", False
    value = smiles
    if ignore_cistrans:
        value = value.replace("/", "").replace("\\", "")
    if Chem.MolFromSmiles(value) is None:
        if not replace_rgroup:
            return smiles, False
        try:
            from SmilesPE.pretokenizer import atomwise_tokenizer

            tokens = atomwise_tokenizer(value)
            for i, token in enumerate(tokens):
                if token.startswith("[") and token.endswith("]"):
                    symbol = token[1:-1]
                    if symbol.startswith("R") and symbol[1:].isdigit():
                        tokens[i] = f"[{symbol[1:]}*]"
                    elif Chem.AtomFromSmiles(token) is None:
                        tokens[i] = "*"
            value = "".join(tokens)
        except Exception:
            return smiles, False
        if Chem.MolFromSmiles(value) is None:
            return smiles, False
    try:
        return Chem.CanonSmiles(value, useChiral=not ignore_chiral), True
    except Exception:
        return smiles, False


def tanimoto(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.0
    try:
        m1, m2 = Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)
        if m1 is None or m2 is None:
            return 0.0
        return float(DataStructs.FingerprintSimilarity(
            Chem.RDKFingerprint(m1), Chem.RDKFingerprint(m2)))
    except Exception:
        return 0.0


def score_smiles(records: Iterable[dict], predictions: Dict[str, Optional[str]]
                 ) -> tuple[dict, list[dict]]:
    """Score records with a ``gt`` SMILES field."""
    rows = []
    for record in records:
        image = record["images"][0]
        gt = record["gt"]
        pred = predictions.get(image)
        gt_s, gt_ok = canonicalize(gt, ignore_cistrans=True)
        pred_s, pred_ok = canonicalize(pred, ignore_cistrans=True)
        gt_g, gt_g_ok = canonicalize(gt, ignore_chiral=True, ignore_cistrans=True)
        pred_g, pred_g_ok = canonicalize(pred, ignore_chiral=True, ignore_cistrans=True)
        acc_s = bool(gt_ok and pred_ok and gt_s == pred_s)
        rows.append(dict(
            image=image,
            gt=gt,
            pred=pred,
            exact=bool(pred) and pred == gt,
            acc_s=acc_s,
            acc_g=bool(gt_g_ok and pred_g_ok and gt_g == pred_g),
            tanimoto=tanimoto(gt, pred),
            gt_chiral="@" in gt_s,
            produced=bool(pred),
            valid=bool(pred_ok),
        ))
    chiral = [row for row in rows if row["gt_chiral"]]
    n = len(rows)
    metrics = dict(
        n=n,
        produced=sum(1 for row in rows if row["valid"]),
        produced_nonempty=sum(1 for row in rows if row["produced"]),
        acc_exact=float(np.mean([row["exact"] for row in rows])) if rows else 0.0,
        acc_s=float(np.mean([row["acc_s"] for row in rows])) if rows else 0.0,
        acc_g=float(np.mean([row["acc_g"] for row in rows])) if rows else 0.0,
        acc_c=float(np.mean([row["acc_s"] for row in chiral])) if chiral else -1.0,
        n_chiral=len(chiral),
        tanimoto=float(np.mean([row["tanimoto"] for row in rows])) if rows else 0.0,
    )
    return metrics, rows
