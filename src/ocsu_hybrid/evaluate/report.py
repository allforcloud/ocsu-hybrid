"""Full-test scoring and report writing."""
from __future__ import annotations

import ast
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

from .. import paths
from ..fg.labels import parse_fg
from .fg import score_fg
from .smiles import score_smiles, strip_prefix
from .text import TextScorer, bertscore


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def load_predictions(path: str | Path) -> Dict[str, dict]:
    return {row["image"]: row for row in read_jsonl(path)}


def _as_list(value) -> list:
    """Accept a JSON list or the Python-repr string older runners wrote."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
            return list(parsed) if isinstance(parsed, (list, tuple, set)) else [value]
        except Exception:
            return [value] if value else []
    return list(value or [])


def extract_records(data: list[dict]) -> dict[str, list[dict]]:
    """Split ``test.json`` into the four OCSU tasks with parsed ground truth."""
    buckets = dict(smiles=[], fg=[], iupac=[], description=[])
    for item in data:
        task = item.get("task_name")
        image = item["images"][0]
        content = item["messages"][1]["content"].strip()
        if task == "trans_smiles":
            buckets["smiles"].append(dict(
                images=[image], gt=strip_prefix(content, "The SMILES is")))
        elif task == "struct_cap":
            buckets["fg"].append(dict(images=[image], gt=parse_fg(content)))
        elif task == "trans_iupac":
            buckets["iupac"].append(dict(
                images=[image], gt=strip_prefix(content, "The IUPAC is")))
        elif task == "general_desp":
            buckets["description"].append(dict(images=[image], gt=content))
    return buckets


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().rstrip(".")).lower()


def _bertscore_block(rows: list[dict], model: str, device: Optional[str] = None
                     ) -> dict:
    pairs = [dict(gt=row["gt"], pred=row["pred"]) for row in rows]
    covered = [pair for pair in pairs if (pair["pred"] or "").strip()]
    out = dict(end_to_end=0.0, raw_end_to_end=0.0, conditional=0.0)
    if not pairs or not covered:
        return out
    share = len(covered) / len(pairs)
    out["conditional"] = bertscore(covered, model, rescale=True, device=device)
    # Misses are empty predictions; BERTScore is computed on hits and scaled by
    # coverage so a miss contributes zero, matching the BLEU/ROUGE convention.
    out["end_to_end"] = bertscore(covered, model, rescale=True, device=device) * share
    out["raw_end_to_end"] = bertscore(covered, model, rescale=False, device=device) * share
    return out


def evaluate_predictions(data_dir: str | Path, pred_dir: str | Path,
                         out_path: str | Path | None = None,
                         details_dir: str | Path | None = None,
                         with_bertscore: bool = False,
                         bertscore_model: str | None = None,
                         device: str | None = None,
                         image_filter: Optional[set[str]] = None) -> dict:
    data_dir = Path(data_dir)
    pred_dir = Path(pred_dir)
    data = json.loads((data_dir / "test.json").read_text(encoding="utf-8"))
    tasks = extract_records(data)
    if image_filter is not None:
        tasks = {name: [row for row in rows
                        if row["images"][0] in image_filter]
                 for name, rows in tasks.items()}

    smiles_pred = {k: v.get("pred") for k, v in
                   load_predictions(pred_dir / "smiles.jsonl").items()}
    fg_pred = {k: _as_list(v.get("pred")) for k, v in
               load_predictions(pred_dir / "fg.jsonl").items()}
    iupac_rows_pred = load_predictions(pred_dir / "iupac.jsonl")
    desc_rows_pred = load_predictions(pred_dir / "desc.jsonl")

    smiles_metrics, smiles_rows = score_smiles(tasks["smiles"], smiles_pred)
    fg_metrics, fg_rows = score_fg(tasks["fg"], fg_pred)

    scorer = TextScorer()

    # ------------------------------------------------------------------ IUPAC
    iupac_rows = []
    for record in tasks["iupac"]:
        image = record["images"][0]
        pred_row = iupac_rows_pred.get(image, {})
        pred = pred_row.get("pred")
        iupac_rows.append(dict(image=image, gt=record["gt"], pred=pred,
                               hit=bool(pred), source=pred_row.get("source")))
    covered = [row for row in iupac_rows if row["hit"]]
    exact = sum(1 for row in covered
                if _norm_text(row["pred"]) == _norm_text(row["gt"]))
    iupac_metrics = dict(
        n=len(iupac_rows),
        covered=len(covered),
        coverage=len(covered) / len(iupac_rows) if iupac_rows else 0.0,
        exact_on_covered=exact / len(covered) if covered else 0.0,
        exact_end_to_end=exact / len(iupac_rows) if iupac_rows else 0.0,
        sources=dict(Counter((row["source"] or "null") if row["hit"] else "null"
                             for row in iupac_rows)),
        conditional=scorer.evaluate(
            [dict(gt=row["gt"], pred=row["pred"]) for row in covered]) if covered else {},
        end_to_end=scorer.evaluate(
            [dict(gt=row["gt"], pred=row["pred"] or "") for row in iupac_rows]),
    )

    # -------------------------------------------------------------- description
    desc_rows = []
    for record in tasks["description"]:
        image = record["images"][0]
        pred = desc_rows_pred.get(image, {}).get("pred")
        desc_rows.append(dict(image=image, gt=record["gt"], pred=pred, hit=bool(pred)))
    desc_covered = [row for row in desc_rows if row["hit"]]
    desc_metrics = dict(
        n=len(desc_rows),
        covered=len(desc_covered),
        coverage=len(desc_covered) / len(desc_rows) if desc_rows else 0.0,
        conditional=scorer.evaluate(
            [dict(gt=row["gt"], pred=row["pred"]) for row in desc_covered]
        ) if desc_covered else {},
        end_to_end=scorer.evaluate(
            [dict(gt=row["gt"], pred=row["pred"] or "") for row in desc_rows]),
    )

    if with_bertscore:
        if not bertscore_model:
            local = paths.models_dir() / "roberta-large"
            bertscore_model = str(local) if local.exists() else None
        model = bertscore_model or "roberta-large"
        iupac_metrics["bertscore"] = _bertscore_block(iupac_rows, model, device)
        pairs = [dict(gt=row["gt"], pred=row["pred"]) for row in desc_rows]
        covered_pairs = [pair for pair in pairs if (pair["pred"] or "").strip()]
        share = len(covered_pairs) / len(pairs) if pairs else 0.0
        desc_metrics["bertscore"] = dict(
            end_to_end=bertscore(covered_pairs, model, rescale=True,
                                 device=device) * share,
            raw_end_to_end=bertscore(covered_pairs, model, rescale=False,
                                     device=device) * share,
        ) if covered_pairs else {}

    result = dict(smiles=smiles_metrics, fg=fg_metrics,
                  iupac=iupac_metrics, description=desc_metrics)

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    if details_dir:
        det = Path(details_dir)
        det.mkdir(parents=True, exist_ok=True)
        for name, rows in (("smiles", smiles_rows), ("fg", fg_rows),
                           ("iupac", iupac_rows), ("desc", desc_rows)):
            (det / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
                encoding="utf-8")
    return result


def format_summary(metrics: dict) -> str:
    """Compact human-readable summary for the CLI."""
    s, f, i, d = (metrics.get("smiles", {}), metrics.get("fg", {}),
                  metrics.get("iupac", {}), metrics.get("description", {}))
    lines = [
        "SMILES      Acc_s {:.2f}  Acc_g {:.2f}  Acc_c {:.2f}  Tanimoto {:.2f}".format(
            100 * s.get("acc_s", 0), 100 * s.get("acc_g", 0),
            100 * s.get("acc_c", 0), 100 * s.get("tanimoto", 0)),
        "FG          F1 {:.2f}  P {:.2f}  R {:.2f}  exact {:.2f}".format(
            100 * f.get("f1", 0), 100 * f.get("precision", 0),
            100 * f.get("recall", 0), 100 * f.get("exact", 0)),
        "IUPAC       coverage {:.2f}%  BL-4 {:.2f}  R-1 {:.2f}  METEOR {:.2f}".format(
            100 * i.get("coverage", 0),
            100 * i.get("end_to_end", {}).get("BLEU-4", 0),
            100 * i.get("end_to_end", {}).get("ROUGE-1", 0),
            100 * i.get("end_to_end", {}).get("METEOR", 0)),
        "Description BL-4 {:.2f}  R-2 {:.2f}  R-L {:.2f}  METEOR {:.2f}".format(
            100 * d.get("end_to_end", {}).get("BLEU-4", 0),
            100 * d.get("end_to_end", {}).get("ROUGE-2", 0),
            100 * d.get("end_to_end", {}).get("ROUGE-L", 0),
            100 * d.get("end_to_end", {}).get("METEOR", 0)),
    ]
    return "\n".join(lines)
