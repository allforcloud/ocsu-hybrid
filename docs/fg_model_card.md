---
license: apache-2.0
tags:
- chemistry
- functional-groups
- convnext
- image-classification
datasets:
- PharMolix/Vis-CheBI20
metrics:
- f1
---

# OCSU-Hybrid functional-group head (ConvNeXt-T)

Multi-label image classifier for the 140 functional-group names used by the OCSU
paper. It replaces the `OCSR -> RDKit dictionary` stage with an image-only head,
so functional groups do not depend on SMILES quality.

## Model

| item | value |
|---|---|
| architecture | ConvNeXt-T (timm), ImageNet initialisation |
| parameters | 27.8 M |
| input | 384 x 384 RGB |
| output | 140 sigmoid scores |
| checkpoint size | 107 MB |
| threshold | 0.72 (tuned on 500 held-out images) |

The checkpoint is a `torch.save` dictionary with:

```python
{
  "model": state_dict,
  "vocab": [140 group names],
  "img_size": 384,
  "threshold": 0.72,
  "val_f1": 0.9944,
}
```

## Training

| item | value |
|---|---|
| data | Vis-CheBI20 train split, 25,644 images |
| labels | parsed from the OCSU `struct_cap` captions |
| loss | `BCEWithLogitsLoss`, `pos_weight = sqrt(neg/pos)` clipped to 1–20 |
| optimiser | AdamW, lr 2e-4, cosine schedule, weight decay 0.05 |
| batch / epochs | 32 / 12 (early stopping, patience 3) |
| augmentation | RandomResizedCrop(0.7–1.0), horizontal flip, colour jitter |
| hardware | RTX 4060 8 GB, about 30 minutes |

## Metrics

Full Vis-CheBI20 test split (3,269 samples, molecule-disjoint from train):

| metric | value |
|---|---:|
| F1 | **99.07** |
| precision | 99.31 |
| recall | 99.00 |
| exact set match | 92.87 |
| macro F1 (117 classes with support) | 87.46 |

Comparison (functional-group F1):

| method | F1 |
|---|---:|
| paper MolScribe + RDKit | 89.60 |
| this dictionary + MolScribe | 91.11 |
| paper DoubleCheck + RDKit | 93.63 |
| paper Mol-VL-7B | 97.32 |
| **this head** | **99.07** |

## Intended use and limits

- The label space is the OCSU dictionary, not free-form chemical naming.
- Training and test images come from the same Vis-CheBI20 rendering pipeline.
  Scanned papers, photographs or screenshots may need domain adaptation.
- The head predicts labels for the whole molecule; it does not localise them.

## Usage

```bash
python scripts/download_models.py --fg \
  --fg-repo allforcloud9527/ocsu-hybrid-fg-convnext-t
ocsu analyze molecule.png
```

or, with the checkpoint path explicit:

```bash
ocsu analyze molecule.png --fg-ckpt /path/to/model.pt
```

## License

Apache-2.0. The Vis-CheBI20 dataset is Apache-2.0; see the repository NOTICE.
