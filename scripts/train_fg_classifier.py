#!/usr/bin/env python3
"""Train the ConvNeXt-T functional-group head.

Input is the OCSU ``train.json``. The labels are the parsed functional-group
captions (140 classes in Vis-CheBI20); the global threshold is tuned on a held-out
500-image split and stored in the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ocsu_hybrid import paths  # noqa: E402
from ocsu_hybrid.fg.labels import mean_f1, parse_fg  # noqa: E402

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FGDataset(Dataset):
    def __init__(self, records, root, vocab, transform):
        self.records = records
        self.root = Path(root)
        self.vocab = vocab
        self.index = {name: i for i, name in enumerate(vocab)}
        self.transform = transform
        self.labels = []
        for record in records:
            target = torch.zeros(len(vocab))
            for name in parse_fg(record["messages"][1]["content"].strip()):
                index = self.index.get(name)
                if index is not None:
                    target[index] = 1.0
            self.labels.append(target)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        with Image.open(self.root / record["images"][0]) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, self.labels[index]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", default=str(paths.data_dir() / "train.json"))
    ap.add_argument("--images-root", default=str(paths.data_dir()))
    ap.add_argument("--out", default=str(paths.models_dir() / "fg_classifier"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--img-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import timm
    from torchvision import transforms as T

    torch.manual_seed(0)
    random.seed(0)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        print("[warn] training on CPU will be slow", flush=True)

    records = [r for r in json.loads(Path(args.train_json).read_text(encoding="utf-8"))
               if r["task_name"] == "struct_cap"]
    random.shuffle(records)
    val, train = records[:args.val_size], records[args.val_size:]
    vocab = sorted({name for r in records
                    for name in parse_fg(r["messages"][1]["content"].strip())})
    print(f"[data] train {len(train)} val {len(val)} classes {len(vocab)}", flush=True)

    train_tf = T.Compose([
        T.RandomResizedCrop(args.img_size, scale=(0.7, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.1, 0.1, 0.1, 0.02),
        T.ToTensor(), T.Normalize(MEAN, STD)])
    val_tf = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(), T.Normalize(MEAN, STD)])

    ds_train = FGDataset(train, args.images_root, vocab, train_tf)
    ds_val = FGDataset(val, args.images_root, vocab, val_tf)
    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=True,
                          num_workers=args.workers, pin_memory=True, drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    pos = torch.stack(ds_train.labels).sum(0)
    neg = len(ds_train) - pos
    pos_weight = torch.clamp((neg / pos.clamp(min=1)).sqrt(), 1.0, 20.0)
    print(f"[data] pos_weight min/median/max {pos_weight.min():.1f}/"
          f"{pos_weight.median():.1f}/{pos_weight.max():.1f}", flush=True)

    model = timm.create_model("convnext_tiny", pretrained=True,
                              num_classes=len(vocab), drop_path_rate=0.1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    steps_per_epoch = len(dl_train)
    total = steps_per_epoch * args.epochs
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = max(1, int(total * 0.03))

    def lr_at(step):
        if step < warmup:
            return args.lr * step / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return 0.5 * args.lr * (1 + math.cos(math.pi * progress))

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    best, bad, step = -1.0, 0, 0
    Path(args.out).mkdir(parents=True, exist_ok=True)
    amp = device.type == "cuda"

    for epoch in range(args.epochs):
        model.train()
        started, running = time.time(), 0.0
        for i, (x, y) in enumerate(dl_train):
            for group in opt.param_groups:
                group["lr"] = lr_at(step)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                loss = criterion(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item()
            step += 1
            if step % 50 == 0:
                print(f"[train] e{epoch} {i}/{steps_per_epoch} loss {running/50:.4f} "
                      f"lr {lr_at(step):.1e} {(time.time()-started)/step:.2f}s/step",
                      flush=True)
                running = 0.0
            if args.max_steps and step >= args.max_steps:
                break

        model.eval()
        probs = []
        with torch.inference_mode():
            for x, _ in dl_val:
                x = x.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    probs.append(torch.sigmoid(model(x)).float().cpu())
        probs = torch.cat(probs).numpy()
        gts = [sorted({n for n in parse_fg(r["messages"][1]["content"].strip())})
               for r in val]
        best_thr, best_f1 = 0.5, -1.0
        for thr in np.arange(0.10, 0.91, 0.02):
            preds = [[vocab[j] for j in range(len(vocab)) if probs[i, j] >= thr]
                     for i in range(len(val))]
            f1 = mean_f1(gts, preds)["f1"]
            if f1 > best_f1:
                best_thr, best_f1 = float(thr), f1
        print(f"[val] epoch {epoch} F1 {best_f1:.4f} thr {best_thr:.2f}", flush=True)

        if best_f1 > best:
            best, bad = best_f1, 0
            torch.save(dict(model=model.state_dict(), vocab=vocab,
                            img_size=args.img_size, threshold=best_thr,
                            val_f1=best_f1), Path(args.out) / "model.pt")
            print(f"[save] best {best:.4f} thr {best_thr:.2f}", flush=True)
        else:
            bad += 1
            if bad >= args.patience:
                print("[stop] early stopping", flush=True)
                break
        if args.max_steps and step >= args.max_steps:
            break

    print(f"[done] best val F1 {best:.4f} -> {Path(args.out) / 'model.pt'}",
          flush=True)


if __name__ == "__main__":
    main()
