#!/usr/bin/env python3
"""Persistent MolScribe worker: load the model once, process a list of images.

The upstream ``MolScribe`` interface pulls in the albumentations-based transform
stack. Inference only needs the Swin encoder, the graph decoder and a handful of
deterministic preprocessing steps, so this worker rebuilds them directly and
reproduces the official preprocessing:

    crop white border (pad 5) -> resize 384 -> gray -> ImageNet normalize

Output JSONL::

    {"image": "...", "smiles": "...", "raw": "...", "error": null, "latency_s": 0.2}
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


def preprocess(path: Path, size: int = 384) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"cannot read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_white(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = np.stack([gray] * 3, -1).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    return img.transpose(2, 0, 1)


def predict_batch(swin, decoder, paths, device):
    arrays = [preprocess(Path(p)) for p in paths]
    x = torch.from_numpy(np.stack(arrays)).float().to(device)
    with torch.inference_mode():
        feats = forward_swin(swin, x)
        preds = decoder.decode(feats, None)
    rgb = [cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB) for p in paths]
    smiles, _, _ = convert_graph_to_smiles(
        [p["chartok_coords"]["coords"] for p in preds],
        [p["chartok_coords"]["symbols"] for p in preds],
        [p["edges"] for p in preds], images=rgb, num_workers=1)
    return preds, smiles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    images = json.loads(Path(args.images_json).read_text())["images"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    swin, decoder = load_model(Path(args.ckpt), device)

    with open(args.out, "w", encoding="utf-8") as f:
        for i in range(0, len(images), args.batch_size):
            chunk = images[i:i + args.batch_size]
            t0 = time.time()
            try:
                preds, smiles = predict_batch(swin, decoder, chunk, device)
                dt = time.time() - t0
                for p, pred, smi in zip(chunk, preds, smiles):
                    raw = pred["chartok_coords"]["smiles"]
                    valid = bool(smi) and smi != "<invalid>" and Chem.MolFromSmiles(smi) is not None
                    f.write(json.dumps(dict(
                        image=p, smiles=smi if valid else None, raw=raw, error=None,
                        latency_s=round(dt / len(chunk), 3)), ensure_ascii=False) + "\n")
            except Exception as exc:  # noqa: BLE001
                if len(chunk) > 1:
                    print(f"[warn] batch failed at {i}: {type(exc).__name__}: {exc}",
                          file=sys.stderr)
                for p in chunk:
                    try:
                        preds, smiles = predict_batch(swin, decoder, [p], device)
                        smi = smiles[0]
                        valid = bool(smi) and smi != "<invalid>" and Chem.MolFromSmiles(smi) is not None
                        rec = dict(image=p, smiles=smi if valid else None,
                                   raw=preds[0]["chartok_coords"]["smiles"],
                                   error=None, latency_s=None)
                    except Exception as exc1:  # noqa: BLE001
                        rec = dict(image=p, smiles=None, raw=None,
                                   error=f"{type(exc1).__name__}: {exc1}", latency_s=None)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{min(i + args.batch_size, len(images))}/{len(images)}]",
                  file=sys.stderr)


if __name__ == "__main__":
    main()
