#!/usr/bin/env python3
"""Run the full Vis-CheBI20 test split.

Equivalent to ``ocsu eval``; kept as a script for users who prefer explicit
commands.
"""
from __future__ import annotations

import argparse

from ocsu_hybrid.full_test import run_full_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out", default="results/full_test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--fg-backend", choices=["classifier", "dictionary"],
                    default="classifier")
    ap.add_argument("--fg-ckpt", default=None)
    ap.add_argument("--molscribe-ckpt", default=None)
    ap.add_argument("--molscribe-python", default=None)
    ap.add_argument("--molscribe-views", default="base",
                    help="e.g. 'base,pad:0.08,rot:3' for multi-view TTA")
    ap.add_argument("--device", default=None)
    ap.add_argument("--iupac-workers", type=int, default=4)
    ap.add_argument("--iupac-cache", default=None)
    ap.add_argument("--biot5-path", default=None)
    ap.add_argument("--no-description", action="store_true")
    ap.add_argument("--with-bertscore", action="store_true")
    ap.add_argument("--bertscore-model", default=None)
    args = ap.parse_args()
    run_full_test(
        args.data_dir, args.out, limit=args.limit, batch_size=args.batch_size,
        fg_backend=args.fg_backend, fg_ckpt=args.fg_ckpt,
        molscribe_ckpt=args.molscribe_ckpt,
        molscribe_python=args.molscribe_python,
        molscribe_views=args.molscribe_views, device=args.device,
        iupac_workers=args.iupac_workers, iupac_cache=args.iupac_cache,
        biot5_path=args.biot5_path, use_description=not args.no_description,
        with_bertscore=args.with_bertscore,
        bertscore_model=args.bertscore_model)


if __name__ == "__main__":
    main()
