# Negative results

These routes were tested on the full Vis-CheBI20 test split or on a random
100-image subset. They are documented because they look attractive on paper but
did not work in practice. None of them is part of the shipped pipeline.

## 1. Mol-VL-7B cannot do image → SMILES

The paper's OCSR-free route uses Mol-VL-7B end to end. It does not report SMILES
accuracy for it, so we measured the released `PharMolix/Mol-VL-7B` directly.

Setup: 100 random test images, NF4 4-bit, 512 px, greedy decoding, 256 new
tokens. The release needs three fixes to run on 8 GB:

1. its `preprocessor_config.json` ships CLIP mean/std, but the model expects
   Qwen2-VL's 0.5/0.5;
2. the tokenizer has no chat template, so the official Qwen2-VL template is
   installed at runtime;
3. prefill logits are trimmed to the last position to save about 200 MB.

Result:

| method (same 100 images) | valid output | Acc_s | Acc_g | Tanimoto |
|---|---:|---:|---:|---:|
| Mol-VL-7B | 85 / 100 | **0.00** | **0.00** | **0.177** |
| MolScribe | 95 / 100 | **0.81** | 0.85 | 0.929 |

The 7B outputs strings that look like SMILES but have the wrong structure:
a sugar predicted as a terpene, butenoic acid predicted as `C`, a cyclohexanol
missing a hydroxyl. This is why the paper does not report its SMILES score.

**Conclusion:** Mol-VL-7B cannot replace MolScribe as the OCSR stage. The public
OCSR ceiling remains MolScribe at 84.00 Acc_s on the full split.

## 2. Mol-VL-7B as an IUPAC fallback: metric-only

The 329 molecules where PubChem found no record were sent to the same 7B model
with an IUPAC prompt. The generated names were checked with OPSIN against the
ground-truth structure.

| metric | value |
|---|---:|
| generated names | 329 / 329 |
| exact name | **0 / 329** |
| OPSIN-verified against GT structure | **0 / 329** |
| OPSIN parsed but different structure | 94 |
| OPSIN unparsable | 235 |
| BLEU-4 on the fallback subset | 21.5 |

Hybrid (retrieval hits + generated fallback):

| setting | coverage | exact name | BLEU-4 | ROUGE-1 | METEOR |
|---|---:|---:|---:|---:|---:|
| retrieval only | 87.72% | 80.0% | 65.98 | 85.39 | 85.41 |
| + 7B fallback | **100.00%** | 80.0% | **77.40** | **89.13** | **88.62** |
| paper Mol-VL-7B | — | — | 76.95 | 81.88 | 81.72 |

The fallback lifts the text metrics to the paper's level but adds **zero**
correct IUPAC names. The gain is n-gram overlap from plausible names, not new
chemical knowledge. Reporting it as an IUPAC result would be misleading, so it is
not shipped.

**Conclusion:** keep retrieved records as the correctness number. A generated
fallback is a metric-only variant.

## 3. MolScribe fine-tuning collapses

Goal: fine-tune the public MolScribe checkpoint on the 25.9k Vis-CheBI20 training
images to push Acc_s from 84.0 to 85+, which would also raise IUPAC coverage.

| item | value |
|---|---|
| initialisation | `swin_base_char_aux_1m680k.pth` |
| data | 25,900 train / 500 validation (7 coordinate-NaN samples removed) |
| targets | `chartok_coords` + `edges`, CoordGen coordinates |
| input | 384², AMP fp16, batch 4 |
| learning rate | encoder 1e-5 / decoder 1e-4, cosine |
| epochs | 2 (about 23 minutes each on a 4060) |
| augmentation | safe rotation, colour jitter |

CoordGen was required to make the pseudo-coordinates match the dataset: RDKit's
default layout had median IoU 0.067, CoordGen 0.896.

Result:

| metric | original checkpoint | fine-tuned epoch 1 | fine-tuned epoch 2 |
|---|---:|---:|---:|
| validation `post_smiles` | ≈ 0.80 | **0.254** | **0.258** |
| validation Tanimoto | — | 0.630 | 0.679 |
| validation `post_graph` | — | 0.436 | 0.458 |
| train loss | — | 2.84 → 0.42 | 0.39 |

The first epoch already collapsed, so this is not overfitting: the training loss
falls while validation quality drops. The decoder learning rate is too high for a
26k-image fine-tune, and the model forgets the pretrained decoding behaviour.

**Conclusion:** keep the public checkpoint. A conservative decoder-only run might
hold the baseline, but it is unlikely to add accuracy.

## 4. MolSight fine-tuning collapses too

MolSight (66.8M) was the other lightweight OCSR candidate. It was fine-tuned for
one epoch on the same 25.9k images.

| item | value |
|---|---|
| initialisation | `pubchem_uspto_smiles_edges_30.pth` |
| data | 25,900 train / 500 validation |
| format | `char,edges`, 512², AMP |
| batch / accumulation | 4 / 8 (batch 8 OOMs on 8 GB) |
| learning rate | encoder 1e-5 / decoder 1e-4 |
| time | about 21 minutes |

Result on the full test split:

| metric | original | fine-tuned | change |
|---|---:|---:|---:|
| Acc_s | **69.58** | 65.12 | −4.46 |
| Acc_g | **78.97** | 75.73 | −3.24 |
| Acc_c | **59.01** | 50.79 | −8.22 |
| Tanimoto | **91.51** | 89.60 | −1.91 |
| empty outputs | 157 | 248 | +91 |

**Conclusion:** same failure mode as MolScribe. Small-data full-parameter
fine-tuning destroys the pretrained decoder.

## 5. MolScribe beam search is not implemented

The vendored MolScribe decoder exposes a `beam_size` argument but its docstring
says "Beam search is not supported yet." A 100-image probe with `beam=4` produced
zero valid outputs and an `IndexError` for every image. The greedy baseline on the
same subset is Acc_s 0.81, so this is a code path problem, not a data problem.

**Conclusion:** no beam-search gains are available from the public code.

## 6. A 2B VLM is not competitive for functional groups

A Qwen2-VL-2B model was fine-tuned for one epoch on 25k functional-group samples
on a single 4060. It reached **59.19 F1**, versus **91.11** for the RDKit
dictionary + MolScribe and **99.07** for the ConvNeXt-T head.

The comparison is not compute-matched: the paper's 2B model used 4×A100 for 50
epochs on 131k samples, roughly two orders of magnitude more compute. But the
result rules out the cheap version of the idea, and the ConvNeXt-T head already
beats the paper's 7B functional-group score (99.07 vs 97.32).

**Conclusion:** use the image-only ConvNeXt-T head for functional groups.

## What this means

| question | answer |
|---|---|
| Can a 7B VLM replace MolScribe? | No: 0.00 Acc_s on 100 images. |
| Can a 7B VLM fill IUPAC misses? | No: 0/329 structurally correct names. |
| Can fine-tuning lift the OCSR ceiling? | Not with 26k images and full-parameter updates; two models collapsed. |
| Is there a cheap decoding gain? | No: beam search is unimplemented. |
| Is a small VLM better at functional groups? | No: 59.19 vs 99.07 for the ConvNeXt-T head. |

The practical conclusion is that the public OCSR ceiling on this benchmark is
MolScribe at **84.00 Acc_s**, and the IUPAC gap is a coverage gap caused by that
ceiling. A stronger OCSR checkpoint — not more generation, more decoding tricks or
a larger fallback model — is the only route that would move the IUPAC number
materially.
