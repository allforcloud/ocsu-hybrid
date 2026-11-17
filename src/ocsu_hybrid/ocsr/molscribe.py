"""MolScribe image -> SMILES adapter.

MolScribe is vendored under ``third_party/MolScribe`` and runs in a persistent
worker process, so a batch pays the model-loading cost once. The worker rebuilds
the official inference preprocessing with OpenCV and therefore does not need the
albumentations stack used by MolScribe's training code.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .. import paths

WORKER = paths.molscribe_worker()


def _worker_env() -> dict:
    env = os.environ.copy()
    env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    env.setdefault("PYTHONWARNINGS", "ignore")
    return env


def recognize_batch(images: Iterable[str | Path], *, ckpt: str | Path | None = None,
                    python: Optional[str] = None, batch_size: int = 8,
                    timeout: int = 7200) -> List[Dict[str, Any]]:
    """Recognize all images with one model load.

    Returns one dict per image with ``image``, ``smiles``, ``raw``, ``error`` and
    ``latency_s``. ``smiles`` is ``None`` when the prediction is not RDKit-valid.
    """
    py = python or os.environ.get("MOLSCRIBE_PY") or sys.executable
    checkpoint = Path(ckpt) if ckpt else paths.molscribe_ckpt()
    paths.require(
        checkpoint,
        "Download MolScribe with `python scripts/download_models.py --molscribe` "
        "or pass --molscribe-ckpt / set MOLSCRIBE_CKPT.",
    )
    if not WORKER.exists():
        raise FileNotFoundError(f"MolScribe worker missing: {WORKER}")
    paths.require(
        paths.molscribe_src() / "molscribe" / "model.py",
        "The vendored MolScribe source is missing. Run from a full source "
        "checkout, or set MOLSCRIBE_SRC.",
    )

    imgs = [str(Path(x).resolve()) for x in images]
    if not imgs:
        return []
    with tempfile.TemporaryDirectory(prefix="ocsu-hybrid-") as td:
        inp = Path(td) / "images.json"
        out = Path(td) / "result.jsonl"
        inp.write_text(json.dumps({"images": imgs}), encoding="utf-8")
        proc = subprocess.run(
            [py, str(WORKER), "--images-json", str(inp), "--out", str(out),
             "--ckpt", str(checkpoint), "--batch-size", str(batch_size)],
            capture_output=True, text=True, timeout=timeout, env=_worker_env(),
            check=False,
        )
        if proc.returncode != 0 or not out.exists():
            tail = (proc.stderr or proc.stdout or "")[-3000:]
            raise RuntimeError(f"MolScribe worker failed ({proc.returncode}):\n{tail}")
        return [json.loads(line) for line in
                out.read_text(encoding="utf-8").splitlines() if line.strip()]


def recognize(image: str | Path, **kwargs) -> Dict[str, Any]:
    """Recognize one image."""
    return recognize_batch([image], **kwargs)[0]


class MolScribeOCR:
    """Small stateful facade used by the pipeline and the CLI."""

    def __init__(self, ckpt: str | Path | None = None, python: str | None = None,
                 batch_size: int = 8):
        self.ckpt = ckpt
        self.python = python
        self.batch_size = batch_size

    def recognize_batch(self, images: Iterable[str | Path], **kwargs):
        options = dict(ckpt=self.ckpt, python=self.python,
                       batch_size=self.batch_size)
        options.update(kwargs)
        return recognize_batch(images, **options)

    def recognize(self, image: str | Path):
        return self.recognize_batch([image])[0]
