"""Filesystem layout and environment overrides.

Everything is resolved lazily so that a user can point the pipeline at a model
cache without editing code:

    OCSU_HOME       repository root (defaults to the checkout containing this file)
    OCSU_MODELS     model directory (default: <root>/models)
    OCSU_DATA       Vis-CheBI20 directory (default: <root>/data/vischebi20)
    OCSU_CACHE      runtime cache (default: <root>/.cache)
    OCSU_FG_CKPT    functional-group checkpoint (default: <models>/fg_classifier/model.pt)
    MOLSCRIBE_CKPT  MolScribe checkpoint (default: <models>/molscribe/swin_base_char_aux_1m680k.pth)
    MOLSCRIBE_SRC   vendored MolScribe source (default: <root>/third_party/MolScribe)
    BIOT5_PATH      BioT5+ directory (default: <models>/biot5-plus-chebi20)
    OCSU_OPSIN_JAR  OPSIN CLI jar (default: <models>/opsin/opsin-cli.jar)
"""
from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get("OCSU_HOME", PACKAGE_DIR.parents[1])).expanduser().resolve()


def _path(env: str, default: Path) -> Path:
    value = os.environ.get(env)
    return Path(value).expanduser().resolve() if value else default


def models_dir() -> Path:
    return _path("OCSU_MODELS", PROJECT_ROOT / "models")


def data_dir() -> Path:
    return _path("OCSU_DATA", PROJECT_ROOT / "data" / "vischebi20")


def cache_dir() -> Path:
    return _path("OCSU_CACHE", PROJECT_ROOT / ".cache")


def molscribe_src() -> Path:
    return _path("MOLSCRIBE_SRC", PROJECT_ROOT / "third_party" / "MolScribe")


def molscribe_ckpt() -> Path:
    return _path(
        "MOLSCRIBE_CKPT",
        models_dir() / "molscribe" / "swin_base_char_aux_1m680k.pth",
    )


def fg_ckpt() -> Path:
    return _path("OCSU_FG_CKPT", models_dir() / "fg_classifier" / "model.pt")


def biot5_path() -> Path:
    return _path("BIOT5_PATH", models_dir() / "biot5-plus-chebi20")


def opsin_jar() -> Path:
    return _path("OCSU_OPSIN_JAR", models_dir() / "opsin" / "opsin-cli.jar")


def molscribe_worker() -> Path:
    return PROJECT_ROOT / "tools" / "molscribe_worker.py"


def third_party_dir() -> Path:
    return PROJECT_ROOT / "third_party"


def scibert_vocab() -> Path:
    return PACKAGE_DIR / "assets" / "scibert" / "vocab.txt"


def require(path: Path, hint: str) -> Path:
    """Raise an actionable error instead of a stack trace deep in a loader."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found.\n{hint}")
    return Path(path)
