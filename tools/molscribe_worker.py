#!/usr/bin/env python3
"""Persistent MolScribe worker: load the model once, process a list of images.

The upstream ``MolScribe`` interface pulls in the albumentations-based transform
stack. Inference only needs the Swin encoder, the graph decoder and a handful of
deterministic preprocessing steps, so this worker rebuilds them directly and
reproduces the official preprocessing:

    crop white border (pad 5) -> resize 384 -> gray -> ImageNet normalize

Multi-view mode (``--views``) runs several deterministic views of the same image
and keeps the best valid candidate. ``--dump-candidates`` returns every view so
a downstream stage (for example the functional-group head) can select one.
This is plain test-time augmentation; no extra weights are involved.

Output JSONL::

    {"image": "...", "smiles": "...", "raw": "...", "error": null,
     "confidence": 0.91, "view": "base", "latency_s": 0.2}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MOLSCRIBE_SRC = Path(os.environ.get(
    "MOLSCRIBE_SRC", ROOT / "third_party" / "MolScribe")).expanduser().resolve()
ONMT_SRC = Path(os.environ.get(
    "ONMT_SRC", ROOT / "third_party")).expanduser().resolve()
# ``ONMT_SRC`` contains a minimal OpenNMT-py 2.2.0 package; MolScribe's
# transformer decoder imports ``onmt.*`` but its __init__ must not pull torchtext.
sys.path.insert(0, str(ONMT_SRC))
sys.path.insert(0, str(MOLSCRIBE_SRC))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402

RDLogger.DisableLog("rdApp.*")

from molscribe.chemistry import convert_graph_to_smiles  # noqa: E402
from molscribe.model import Decoder  # noqa: E402
from molscribe.tokenizer import get_tokenizer  # noqa: E402
from molscribe.transformer.swin_transformer import SwinTransformer  # noqa: E402

DEFAULT_CKPT = Path(os.environ.get(
    "MOLSCRIBE_CKPT",
    ROOT / "models" / "molscribe" / "swin_base_char_aux_1m680k.pth"))
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def build_args(states):
    args = argparse.Namespace(
        encoder="swin_base", decoder="transformer", trunc_encoder=False,
        no_pretrained=True, use_checkpoint=False, dropout=0.5, embed_dim=256,
        enc_pos_emb=False, dec_num_layers=6, dec_hidden_size=256, dec_attn_heads=8,
        dec_num_queries=128, hidden_dropout=0.1, attn_dropout=0.1,
        max_relative_positions=0, continuous_coords=False, compute_confidence=False,
        input_size=384, vocab_file=None, coord_bins=64, sep_xy=True,
        formats=["chartok_coords", "edges"],
    )
    args.__dict__.update(states.get("args") or {})
    args.encoder_dim = 1024
    return args


def load_model(ckpt: Path, device):
    states = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    args = build_args(states)

    swin = SwinTransformer(
        img_size=384, patch_size=4, in_chans=3, num_classes=1000, embed_dim=128,
        depths=(2, 2, 18, 2), num_heads=(4, 8, 16, 32), window_size=12, mlp_ratio=4.0,
        qkv_bias=True, qk_scale=None, drop_rate=0.0, attn_drop_rate=0.0,
        drop_path_rate=0.1, norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False)
    missing, _ = swin.load_state_dict(
        {k.replace("module.transformer.", ""): v for k, v in states["encoder"].items()},
        strict=False)
    if len(missing) > 2:  # only the unused classification head may be missing
        raise RuntimeError(f"encoder weights missing: {missing[:5]}")

    decoder = Decoder(args, get_tokenizer(args))
    decoder.load_state_dict(
        {k.replace("module.", ""): v for k, v in states["decoder"].items()}, strict=True)

    swin.to(device).eval()
    decoder.to(device).eval()
    return swin, decoder


def forward_swin(swin, x):
    x, h, w = swin.patch_embed(x)
    if swin.absolute_pos_embed is not None:
        x = x + swin.absolute_pos_embed
    x = swin.pos_drop(x)
    hiddens = []
    for layer in swin.layers:
        x, h, w = layer(x, h, w, hiddens)
    x = swin.norm(x)
    hiddens[-1] = x.view_as(hiddens[-1])
    return x


def crop_white(img: np.ndarray, pad: int = 5, value: int = 255) -> np.ndarray:
    diff = (img != value).sum(axis=2)
    if diff.sum() == 0:
        return img
    rows, cols = diff.sum(axis=1), diff.sum(axis=0)
    top = 0
    while rows[top] == 0 and top + 1 < len(rows):
        top += 1
    bottom = len(rows)
    while rows[bottom - 1] == 0 and bottom - 1 > top:
        bottom -= 1
    left = 0
    while cols[left] == 0 and left + 1 < len(cols):
        left += 1
    right = len(cols)
    while cols[right - 1] == 0 and right - 1 > left:
        right -= 1
    crop = img[top:bottom, left:right]
    return cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                              value=(255, 255, 255))


def parse_view(spec: str):
    """Parse ``base``, ``pad:0.08``, ``rot:3`` or ``zoom:0.04``."""
    spec = spec.strip()
    if spec == "base":
        return ("base", 0.0, "base")
    if ":" not in spec:
        raise ValueError(f"bad view spec: {spec}")
    kind, value = spec.split(":", 1)
    if kind not in {"pad", "rot", "zoom"}:
        raise ValueError(f"bad view kind: {kind}")
    return (kind, float(value), spec.replace(":", ""))


def preprocess(path: Path, size: int = 384, view=("base", 0.0, "base")):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_white(img)
    kind, value, _ = view
    h, w = img.shape[:2]
    if kind == "pad":
        p = max(1, int(min(h, w) * value))
        img = cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT,
                                 value=(255, 255, 255))
    elif kind == "rot":
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), value, 1.0)
        img = cv2.warpAffine(img, matrix, (w, h), borderValue=(255, 255, 255))
    elif kind == "zoom":
        d = max(1, int(min(h, w) * value))
        img = img[d:h - d, d:w - d]
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.stack([gray] * 3, -1).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


def run_view(swin, decoder, images, view, *, batch_size, device, beam_size,
             confidence):
    """Return {image: dict(smiles, raw, confidence, error)} for one view."""
    out = {}
    for i in range(0, len(images), batch_size):
        chunk = images[i:i + batch_size]
        t0 = time.time()
        try:
            arrays = [preprocess(Path(p), view=view) for p in chunk]
            x = torch.from_numpy(np.stack(arrays)).float().to(device)
            with torch.inference_mode():
                feats = forward_swin(swin, x)
                preds = decoder.decode(feats, None, beam_size=beam_size)
            rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in chunk]
            smiles, _, _ = convert_graph_to_smiles(
                [p["chartok_coords"]["coords"] for p in preds],
                [p["chartok_coords"]["symbols"] for p in preds],
                [p["edges"] for p in preds], images=rgb, num_workers=1)
            dt = time.time() - t0
            for p, pred, smi in zip(chunk, preds, smiles):
                valid = bool(smi) and smi != "<invalid>" and Chem.MolFromSmiles(smi) is not None
                out[p] = dict(
                    smiles=smi if valid else None,
                    raw=pred["chartok_coords"]["smiles"],
                    confidence=float(pred.get("overall_score") or 0.0)
                    if confidence else None,
                    error=None if valid else "invalid_smiles",
                    latency_s=round(dt / len(chunk), 3),
                )
        except Exception as exc:  # noqa: BLE001
            for p in chunk:
                try:
                    arr = preprocess(Path(p), view=view)
                    x = torch.from_numpy(arr).float().unsqueeze(0).to(device)
                    with torch.inference_mode():
                        preds = decoder.decode(forward_swin(swin, x), None,
                                               beam_size=beam_size)
                    rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)]
                    smi, _, _ = convert_graph_to_smiles(
                        [preds[0]["chartok_coords"]["coords"]],
                        [preds[0]["chartok_coords"]["symbols"]],
                        [preds[0]["edges"]], images=rgb, num_workers=1)
                    valid = bool(smi) and smi != "<invalid>" and Chem.MolFromSmiles(smi) is not None
                    out[p] = dict(
                        smiles=smi if valid else None,
                        raw=preds[0]["chartok_coords"]["smiles"],
                        confidence=float(preds[0].get("overall_score") or 0.0)
                        if confidence else None,
                        error=None if valid else "invalid_smiles",
                        latency_s=None,
                    )
                except Exception as exc1:  # noqa: BLE001
                    out[p] = dict(smiles=None, raw=None, confidence=None,
                                  error=f"{type(exc1).__name__}: {exc1}",
                                  latency_s=None)
            if len(chunk) > 1:
                print(f"[warn] batch failed at {i}: {type(exc).__name__}: {exc}",
                      file=sys.stderr)
        print(f"[{view[2]}] {min(i + batch_size, len(images))}/{len(images)}",
              file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--beam-size", type=int, default=1)
    ap.add_argument("--views", default="base",
                    help="comma list: base,pad:0.08,rot:3,rot:-3,zoom:0.04")
    ap.add_argument("--confidence", action="store_true",
                    help="store model confidence even in single-view mode")
    ap.add_argument("--dump-candidates", action="store_true",
                    help="include every view candidate in the output")
    args_c = ap.parse_args()

    views = [parse_view(v) for v in args_c.views.split(",") if v.strip()]
    images = json.loads(Path(args_c.images_json).read_text())["images"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin, decoder = load_model(Path(args_c.ckpt), device)
    need_conf = args_c.confidence or len(views) > 1
    decoder.compute_confidence = need_conf

    started = time.time()
    candidates = {p: [] for p in images}
    for view in views:
        result = run_view(swin, decoder, images, view,
                          batch_size=args_c.batch_size, device=device,
                          beam_size=args_c.beam_size, confidence=need_conf)
        for p, rec in result.items():
            if rec["smiles"]:
                candidates[p].append(dict(rec, view=view[2]))

    with open(args_c.out, "w", encoding="utf-8") as f:
        for p in images:
            cands = candidates[p]
            if cands:
                best = max(cands, key=lambda r: r["confidence"] or 0.0)
                rec = dict(image=p, smiles=best["smiles"], raw=best["raw"],
                           error=None, confidence=best["confidence"],
                           view=best["view"],
                           latency_s=round((time.time() - started) / max(len(images), 1), 3))
            else:
                rec = dict(image=p, smiles=None, raw=None,
                           error="no_valid_candidate", confidence=None, view=None,
                           latency_s=round((time.time() - started) / max(len(images), 1), 3))
            if args_c.dump_candidates:
                rec["candidates"] = [
                    dict(view=c["view"], smiles=c["smiles"],
                         confidence=c["confidence"], raw=c.get("raw"))
                    for c in cands
                ]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[done] {len(images)} images, views={[v[2] for v in views]}, "
          f"{time.time() - started:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
