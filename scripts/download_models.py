#!/usr/bin/env python3
"""Download the public checkpoints used by OCSU-Hybrid.

Examples::

    python scripts/download_models.py --all
    python scripts/download_models.py --molscribe --biot5 --opsin
    python scripts/download_models.py --fg --fg-repo your-name/ocsu-hybrid-fg-convnext-t
    python scripts/download_models.py --check
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsu_hybrid import paths  # noqa: E402
from ocsu_hybrid.credentials import hf_credentials  # noqa: E402

MOLSCRIBE_REPO = "yujieq/MolScribe"
MOLSCRIBE_FILE = "swin_base_char_aux_1m680k.pth"
BIOT5_REPO = "QizhiPei/biot5-plus-base-chebi20"
BIOT5_FILES = [
    "config.json", "generation_config.json", "added_tokens.json",
    "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json",
    "spiece.model", "model.safetensors",
]
DEFAULT_FG_REPO = os.environ.get(
    "OCSU_FG_REPO", "allforcloud9527/ocsu-hybrid-fg-convnext-t")
OPSIN_URL = ("https://github.com/dan2097/opsin/releases/download/2.9.0/"
             "opsin-cli-2.9.0-jar-with-dependencies.jar")


def _hf():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise SystemExit("huggingface_hub is missing: pip install -e '.[all]'") from exc
    return hf_hub_download, snapshot_download


def download_molscribe(token: str | None = None) -> Path:
    hf_hub_download, _ = _hf()
    target = paths.models_dir() / "molscribe"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[molscribe] {MOLSCRIBE_REPO}/{MOLSCRIBE_FILE}")
    path = hf_hub_download(MOLSCRIBE_REPO, MOLSCRIBE_FILE,
                           local_dir=str(target), token=token or None)
    print(f"[molscribe] -> {path}")
    return Path(path)


def download_biot5(token: str | None = None) -> Path:
    _, snapshot_download = _hf()
    target = paths.biot5_path()
    print(f"[biot5] {BIOT5_REPO} -> {target}")
    snapshot_download(BIOT5_REPO, local_dir=str(target),
                      allow_patterns=BIOT5_FILES, token=token or None)
    print(f"[biot5] -> {target}")
    return target


def download_fg(repo_id: str, token: str | None = None) -> Path:
    hf_hub_download, _ = _hf()
    if repo_id.startswith("your-org/"):
        print("[fg] The ConvNeXt-T weights are published on Hugging Face, not in git.\n"
              "     Set your repository in configs/local.yaml, with --fg-repo, "
              "or with OCSU_FG_REPO, e.g.\n"
              "       python scripts/download_models.py --fg --fg-repo your-name/ocsu-hybrid-fg-convnext-t")
        return paths.fg_ckpt()
    target = paths.models_dir() / "fg_classifier"
    target.mkdir(parents=True, exist_ok=True)
    print(f"[fg] {repo_id}/model.pt")
    path = hf_hub_download(repo_id, "model.pt", local_dir=str(target),
                           token=token or None)
    print(f"[fg] -> {path}")
    return Path(path)


def download_opsin(url: str = OPSIN_URL) -> Path:
    target = paths.opsin_jar()
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"[opsin] already present: {target}")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[opsin] {url}")
    tmp = target.with_suffix(".jar.part")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while True:
            chunk = resp.read(1024 * 256)
            if not chunk:
                break
            out.write(chunk)
            done += len(chunk)
            if total:
                print(f"\r[opsin] {done / total:6.1%}", end="", flush=True)
    print()
    tmp.replace(target)
    print(f"[opsin] -> {target}")
    return target


def download_nltk() -> None:
    import nltk

    for resource in ("wordnet", "omw-1.4"):
        print(f"[nltk] {resource}")
        nltk.download(resource, quiet=True)


def check() -> int:
    expected = [
        ("MolScribe", paths.molscribe_ckpt()),
        ("BioT5+ config", paths.biot5_path() / "config.json"),
        ("BioT5+ weights", paths.biot5_path() / "model.safetensors"),
        ("FG ConvNeXt-T", paths.fg_ckpt()),
        ("OPSIN", paths.opsin_jar()),
        ("SciBERT vocab", paths.scibert_vocab()),
    ]
    missing = 0
    for name, path in expected:
        exists = Path(path).exists()
        size = Path(path).stat().st_size / 1e6 if exists else 0
        print(f"{'OK ' if exists else 'MISS'}  {name:16s} {path}  {size:.1f} MB")
        missing += not exists
    return 1 if missing else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--molscribe", action="store_true")
    ap.add_argument("--biot5", action="store_true")
    ap.add_argument("--fg", action="store_true")
    ap.add_argument("--opsin", action="store_true")
    ap.add_argument("--nltk", action="store_true")
    ap.add_argument("--bertscore", action="store_true",
                    help="download roberta-large for optional BERTScore")
    ap.add_argument("--fg-repo", default=None)
    ap.add_argument("--opsin-url", default=OPSIN_URL)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        return check()
    if not any([args.all, args.molscribe, args.biot5, args.fg, args.opsin,
                args.nltk, args.bertscore]):
        ap.error("choose --all or at least one component (see --help)")

    try:
        username, token, config_repo = hf_credentials()
    except Exception as exc:  # noqa: BLE001 - a bad config must not block public downloads
        print(f"[config] warning: {exc}", file=sys.stderr)
        username, token, config_repo = None, None, None

    fg_repo = (args.fg_repo or os.environ.get("OCSU_FG_REPO")
               or config_repo or DEFAULT_FG_REPO)
    if "/" not in fg_repo and username:
        fg_repo = f"{username}/{fg_repo}"
    if token:
        print(f"[config] Hugging Face token loaded from "
              f"{'environment' if os.environ.get('HF_TOKEN') else 'configs/local.yaml'}")

    if args.all or args.molscribe:
        download_molscribe(token)
    if args.all or args.biot5:
        download_biot5(token)
    if args.all or args.fg:
        download_fg(fg_repo, token)
    if args.all or args.opsin:
        download_opsin(args.opsin_url)
    if args.all or args.nltk:
        download_nltk()
    if args.bertscore:
        _, snapshot_download = _hf()
        target = paths.models_dir() / "roberta-large"
        print(f"[bertscore] roberta-large -> {target}")
        snapshot_download("roberta-large", local_dir=str(target),
                          token=token or None)

    print("\nRun `python scripts/download_models.py --check` to verify.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
