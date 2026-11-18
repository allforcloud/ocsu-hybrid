"""Full Vis-CheBI20 test runner.

The four tasks share the same images. OCSR runs once per unique image, the
functional-group head runs once per FG sample, and the IUPAC / description
stages consume the same SMILES.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterable, List

from .evaluate.report import evaluate_predictions, format_summary
from .fg.dictionary import get_func_groups
from .pipeline import Pipeline
from .validate import validate_smiles


def _read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _append_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
        f.flush()


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                            for row in rows), encoding="utf-8")


def _select_tta(ocsr_rows: dict, groups_by_image: dict) -> None:
    """Replace each worker-selected SMILES with the FG-consistent view."""
    for image, row in ocsr_rows.items():
        candidates = row.get("candidates") or []
        if not candidates:
            continue
        image_groups = groups_by_image.get(image)
        best = None
        best_score = None
        for candidate in candidates:
            smiles = candidate.get("smiles")
            if not smiles:
                continue
            score = (Pipeline._fg_f1(get_func_groups(smiles), image_groups)
                     if image_groups is not None else 0.0,
                     candidate.get("confidence") or 0.0)
            if best_score is None or score > best_score:
                best_score = score
                best = candidate
        if best is not None:
            row["smiles"] = best["smiles"]
            row["raw"] = best.get("raw", row.get("raw"))
            row["selected_view"] = best.get("view")
            row["selected_confidence"] = best.get("confidence")


def _task_images(records: List[dict], task: str, limit: int) -> List[str]:
    images = [r["images"][0] for r in records if r.get("task_name") == task]
    return images[:limit] if limit else images


def _ordered_unique(values: Iterable[str]) -> List[str]:
    seen, out = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def run_full_test(data_dir: str | Path, out_dir: str | Path, *,
                  limit: int = 0,
                  batch_size: int = 8,
                  fg_backend: str = "classifier",
                  fg_ckpt: str | Path | None = None,
                  molscribe_ckpt: str | Path | None = None,
                  molscribe_python: str | None = None,
                  molscribe_views: str = "base",
                  device: str | None = None,
                  iupac_workers: int = 4,
                  iupac_cache: str | Path | None = None,
                  biot5_path: str | Path | None = None,
                  use_description: bool = True,
                  with_bertscore: bool = False,
                  bertscore_model: str | None = None) -> dict:
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    pred_dir = out_dir / "preds"
    details_dir = out_dir / "details"
    pred_dir.mkdir(parents=True, exist_ok=True)

    records = json.loads((data_dir / "test.json").read_text(encoding="utf-8"))
    smiles_images = _task_images(records, "trans_smiles", limit)
    fg_images = _task_images(records, "struct_cap", limit)
    iupac_images = _task_images(records, "trans_iupac", limit)
    desc_images = _task_images(records, "general_desp", limit)
    needed = _ordered_unique(smiles_images + fg_images + iupac_images + desc_images)
    print(f"[data] smiles {len(smiles_images)}  fg {len(fg_images)}  "
          f"iupac {len(iupac_images)}  desc {len(desc_images)}  "
          f"unique images {len(needed)}", flush=True)

    pipeline = Pipeline(
        molscribe_ckpt=molscribe_ckpt,
        molscribe_python=molscribe_python,
        molscribe_batch_size=batch_size,
        molscribe_views=molscribe_views,
        fg_backend=fg_backend,
        fg_ckpt=fg_ckpt,
        device=device,
        iupac_workers=iupac_workers,
        iupac_cache=iupac_cache,
        biot5_path=biot5_path,
        use_iupac=True,
        use_description=use_description,
    )

    # ------------------------------------------------------------------ OCSR
    ocsr_path = pred_dir / "ocsr.jsonl"
    ocsr_rows = {row["image"]: row for row in _read_jsonl(ocsr_path)}
    missing = [image for image in needed if image not in ocsr_rows]
    print(f"[ocsr] {len(missing)} to run / {len(needed)}", flush=True)
    ocsr_chunk = max(batch_size, 256)  # one model load per chunk, not per GPU batch
    for start in range(0, len(missing), ocsr_chunk):
        chunk = missing[start:start + ocsr_chunk]
        rows = pipeline.recognize([data_dir / image for image in chunk])
        for row, image in zip(rows, chunk):
            row["image"] = image  # keep relative paths from test.json
        _append_jsonl(ocsr_path, rows)
        for row in rows:
            ocsr_rows[row["image"]] = row
        print(f"[ocsr] {min(start + ocsr_chunk, len(missing))}/{len(missing)}",
              flush=True)

    # -------------------------------------------------------- functional groups
    fg_path = pred_dir / "fg.jsonl"
    fg_rows = {row["image"]: row for row in _read_jsonl(fg_path)}
    missing_fg = [image for image in fg_images if image not in fg_rows]

    if fg_backend == "classifier":
        if missing_fg:
            print(f"[fg] classifier on {len(missing_fg)} images", flush=True)
            groups = pipeline.fg().predict(
                [data_dir / image for image in missing_fg], batch_size=batch_size)
            rows = [dict(image=image, pred=pred)
                    for image, pred in zip(missing_fg, groups)]
            _append_jsonl(fg_path, rows)
            for row in rows:
                fg_rows[row["image"]] = row
        groups_by_image = {image: row.get("pred") or []
                           for image, row in fg_rows.items()}
        if molscribe_views != "base":
            print(f"[tta] selecting {len(ocsr_rows)} candidates with FG head",
                  flush=True)
            _select_tta(ocsr_rows, groups_by_image)
    else:
        if molscribe_views != "base":
            print(f"[tta] selecting {len(ocsr_rows)} candidates by confidence",
                  flush=True)
            _select_tta(ocsr_rows, {})

    canonical = {image: validate_smiles(ocsr_rows.get(image, {}).get("smiles"))["canonical"]
                 for image in needed}
    _write_jsonl(ocsr_path, [ocsr_rows[image] for image in needed if image in ocsr_rows])
    smiles_path = pred_dir / "smiles.jsonl"
    _write_jsonl(smiles_path, [
        dict(image=image, pred=ocsr_rows[image].get("smiles"),
             raw=ocsr_rows[image].get("raw"))
        for image in smiles_images if image in ocsr_rows
    ])

    if fg_backend == "dictionary" and missing_fg:
        print(f"[fg] dictionary on {len(missing_fg)} images", flush=True)
        rows = [dict(image=image, pred=get_func_groups(canonical.get(image)))
                for image in missing_fg]
        _append_jsonl(fg_path, rows)
        for row in rows:
            fg_rows[row["image"]] = row
    print(f"[fg] {len(fg_images)} predictions ready", flush=True)

    # ------------------------------------------------------------------ IUPAC
    iupac_path = pred_dir / "iupac.jsonl"
    iupac_rows = {row["image"]: row for row in _read_jsonl(iupac_path)}
    missing_iupac = [image for image in iupac_images if image not in iupac_rows]
    if missing_iupac:
        print(f"[iupac] {len(missing_iupac)} lookups, workers={iupac_workers}",
              flush=True)
        started = time.time()
        results = pipeline.resolver().resolve_many(
            [canonical.get(image) for image in missing_iupac],
            workers=iupac_workers)
        rows = [dict(image=image, smiles=canonical.get(image),
                     pred=name, source=source)
                for image, (name, source) in zip(missing_iupac, results)]
        _append_jsonl(iupac_path, rows)
        for row in rows:
            iupac_rows[row["image"]] = row

        # A TTA view can be the best structural match yet have no PubChem
        # record. Try the base view and the remaining views before giving up.
        misses = [row["image"] for row in rows if not row.get("pred")]
        if misses:
            alt_pairs = []
            for image in misses:
                seen = {canonical.get(image)}
                candidates = sorted(
                    ocsr_rows.get(image, {}).get("candidates") or [],
                    key=lambda c: (c.get("view") != "base",
                                   -(c.get("confidence") or 0.0)))
                for candidate in candidates:
                    smiles = candidate.get("smiles")
                    value = validate_smiles(smiles)["canonical"] if smiles else None
                    if value and value not in seen:
                        seen.add(value)
                        alt_pairs.append((image, value))
            if alt_pairs:
                unique = list(dict.fromkeys(value for _, value in alt_pairs))
                alt_results = dict(zip(
                    unique, pipeline.resolver().resolve_many(
                        unique, workers=iupac_workers)))
                recovered = 0
                for image, value in alt_pairs:
                    if iupac_rows[image].get("pred"):
                        continue
                    name, source = alt_results.get(value, (None, None))
                    if name:
                        canonical[image] = value
                        ocsr_rows[image]["smiles"] = value
                        iupac_rows[image].update(smiles=value, pred=name,
                                                 source=source)
                        recovered += 1
                if recovered:
                    _write_jsonl(iupac_path, [iupac_rows[image] for image in iupac_images
                                              if image in iupac_rows])
                    _write_jsonl(ocsr_path, [ocsr_rows[image] for image in needed
                                             if image in ocsr_rows])
                    _write_jsonl(smiles_path, [
                        dict(image=image, pred=ocsr_rows[image].get("smiles"),
                             raw=ocsr_rows[image].get("raw"))
                        for image in smiles_images if image in ocsr_rows
                    ])
                    print(f"[iupac] recovered {recovered} names from other views",
                          flush=True)
        print(f"[iupac] done in {time.time() - started:.1f}s", flush=True)

    # -------------------------------------------------------------- description
    desc_path = pred_dir / "desc.jsonl"
    if use_description:
        desc_rows = {row["image"]: row for row in _read_jsonl(desc_path)}
        missing_desc = [image for image in desc_images if image not in desc_rows]
        if missing_desc:
            print(f"[desc] BioT5+ on {len(missing_desc)} images", flush=True)
            describer = pipeline.describer()
            for start in range(0, len(missing_desc), 64):
                chunk = missing_desc[start:start + 64]
                texts = describer.describe_many([canonical.get(image) for image in chunk])
                rows = [dict(image=image, smiles=canonical.get(image), pred=text)
                        for image, text in zip(chunk, texts)]
                _append_jsonl(desc_path, rows)
                for row in rows:
                    desc_rows[row["image"]] = row
                print(f"[desc] {min(start + 64, len(missing_desc))}/{len(missing_desc)}",
                      flush=True)
    else:
        print("[desc] skipped", flush=True)

    # ------------------------------------------------------------------- score
    print("[score] evaluating", flush=True)
    metrics = evaluate_predictions(
        data_dir, pred_dir, out_path=out_dir / "metrics.json",
        details_dir=details_dir, with_bertscore=with_bertscore,
        bertscore_model=bertscore_model, device=device,
        image_filter=set(needed))
    (out_dir / "summary.txt").write_text(format_summary(metrics) + "\n",
                                         encoding="utf-8")
    print(format_summary(metrics), flush=True)
    print(f"[done] {out_dir / 'metrics.json'}", flush=True)
    return metrics
