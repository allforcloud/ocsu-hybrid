"""Command-line interface for OCSU-Hybrid."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .full_test import run_full_test
from .pipeline import Pipeline

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def _add_pipeline_args(parser: argparse.ArgumentParser, *,
                       stages: bool = True, verify: bool = True) -> None:
    parser.add_argument("--molscribe-ckpt", default=None,
                        help="MolScribe checkpoint (default: models/molscribe/...)")
    parser.add_argument("--molscribe-python", default=None,
                        help="Python executable for the MolScribe worker "
                             "(default: the current interpreter)")
    parser.add_argument("--fg-backend", choices=["classifier", "dictionary"],
                        default="classifier",
                        help="learned image head (99.07 F1) or RDKit dictionary (91.11)")
    parser.add_argument("--fg-ckpt", default=None,
                        help="functional-group checkpoint (default: models/fg_classifier/model.pt)")
    parser.add_argument("--device", default=None, help="cuda / cpu / cuda:0")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="MolScribe and functional-group batch size")
    parser.add_argument("--iupac-workers", type=int, default=4)
    parser.add_argument("--iupac-cache", default=None,
                        help="PubChem cache JSONL (default: .cache/iupac_pubchem.jsonl)")
    parser.add_argument("--biot5-path", default=None,
                        help="BioT5+ directory (default: models/biot5-plus-chebi20)")
    parser.add_argument("--opsin-jar", default=None)
    if stages:
        parser.add_argument("--no-iupac", action="store_true")
        parser.add_argument("--no-description", action="store_true")
    if verify:
        parser.add_argument("--verify-iupac", action="store_true",
                            help="run OPSIN on every retrieved name (slower)")


def _pipeline(args) -> Pipeline:
    return Pipeline(
        molscribe_ckpt=args.molscribe_ckpt,
        molscribe_python=args.molscribe_python,
        molscribe_batch_size=args.batch_size,
        fg_backend=args.fg_backend,
        fg_ckpt=args.fg_ckpt,
        device=args.device,
        iupac_workers=args.iupac_workers,
        iupac_cache=args.iupac_cache,
        biot5_path=args.biot5_path,
        opsin_jar=args.opsin_jar,
        use_iupac=not args.no_iupac,
        use_description=not args.no_description,
        verify_iupac=args.verify_iupac,
        fg_batch_size=args.batch_size,
    )


def _collect(directory: Path, recursive: bool, limit: int) -> list[Path]:
    iterator = directory.rglob("*") if recursive else directory.iterdir()
    files = sorted(p for p in iterator
                   if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="ocsu",
        description="OCSU-Hybrid: image -> SMILES / functional groups / IUPAC / description")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="analyze one image")
    analyze.add_argument("image")
    analyze.add_argument("--json", default=None, help="also write the result here")
    _add_pipeline_args(analyze)

    batch = sub.add_parser("batch", help="analyze a directory or a list of images")
    batch.add_argument("input", help="image directory, or a text file with one path per line")
    batch.add_argument("-o", "--out", required=True, help="output JSONL")
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--limit", type=int, default=0)
    _add_pipeline_args(batch)

    evaluate = sub.add_parser("eval", help="run the full Vis-CheBI20 test split")
    evaluate.add_argument("--data-dir", required=True,
                          help="Vis-CheBI20 split containing test.json and images")
    evaluate.add_argument("--out", default="results/full_test")
    evaluate.add_argument("--limit", type=int, default=0,
                          help="cap each task (smoke test)")
    evaluate.add_argument("--with-bertscore", action="store_true")
    evaluate.add_argument("--bertscore-model", default=None)
    _add_pipeline_args(evaluate, stages=False, verify=False)

    args = parser.parse_args(argv)

    if args.command == "analyze":
        result = _pipeline(args).analyze(args.image)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.json:
            Path(args.json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.json).write_text(
                json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        return 0

    if args.command == "batch":
        source = Path(args.input)
        if source.is_dir():
            files = _collect(source, args.recursive, args.limit)
        elif source.is_file():
            lines = [line.strip() for line in source.read_text(encoding="utf-8").splitlines()
                     if line.strip() and not line.startswith("#")]
            files = [Path(line) for line in lines]
            if args.limit:
                files = files[:args.limit]
        else:
            raise FileNotFoundError(source)
        if not files:
            print(f"[batch] no images in {source}", file=sys.stderr)
            return 1
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pipeline = _pipeline(args)
        with open(out, "w", encoding="utf-8") as f:
            for start in range(0, len(files), max(1, args.batch_size)):
                chunk = files[start:start + max(1, args.batch_size)]
                results = pipeline.analyze_batch(chunk)
                for result in results:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[batch] {min(start + len(chunk), len(files))}/{len(files)}",
                      flush=True)
        print(f"[done] {len(files)} -> {out}", flush=True)
        return 0

    metrics = run_full_test(
        args.data_dir, args.out, limit=args.limit,
        batch_size=args.batch_size, fg_backend=args.fg_backend,
        fg_ckpt=args.fg_ckpt, molscribe_ckpt=args.molscribe_ckpt,
        molscribe_python=args.molscribe_python, device=args.device,
        iupac_workers=args.iupac_workers, iupac_cache=args.iupac_cache,
        biot5_path=args.biot5_path, with_bertscore=args.with_bertscore,
        bertscore_model=args.bertscore_model)
    return 0 if metrics else 1


if __name__ == "__main__":
    raise SystemExit(main())
