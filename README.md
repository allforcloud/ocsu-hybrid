# OCSU-Hybrid

**English** | [简体中文](README.zh-CN.md)

OCSU-Hybrid is a modular, runnable implementation of the OCSR-based route in
[OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific
Discovery](https://arxiv.org/abs/2501.15415). It turns a 2D molecular structure
image into SMILES, functional groups, an IUPAC name and an English description.
The pipeline combines MolScribe, ConvNeXt-T, PubChem and BioT5+, uses public
specialist models, and fits on a single 8 GB GPU.

**Hugging Face weights:** the functional-group head is published at
[`allforcloud9527/ocsu-hybrid-fg-convnext-t`](https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t).

```
image ──► MolScribe ──► SMILES ──┬──► PubChem ──────► IUPAC name
           (95.5 M)              ├──► BioT5+ ───────► description
                                 │    (252 M)
                                 └──► RDKit dictionary ► functional groups
                                                           (fallback)
          image ───────────────► ConvNeXt-T ─────────► functional groups
                                  (27.8 M, default)
```

---

## Results

Full Vis-CheBI20 test split, same metric definitions as the paper. The paper's
two routes and this project are compared side by side.

| task | metric | **OCSU-Hybrid** | OCSR-based: MolScribe | OCSR-based: DoubleCheck | OCSR-free: Mol-VL-2B | OCSR-free: Mol-VL-7B |
|---|---|---:|---:|---:|---:|---:|
| SMILES | Acc_s | **85.42** | 80.50 | 82.40 | — | — |
| SMILES | Acc_c | **82.07** | 75.10 | 77.60 | — | — |
| SMILES | Tanimoto | **94.28** | 90.80 | 92.70 | — | — |
| functional groups | F1 | **99.07** | 89.60 | 93.63 | 95.40 | 97.32 |
| IUPAC | BLEU-4 | **71.97** | 61.03 | 64.23 | 72.79 | **76.95** |
| IUPAC | ROUGE-1 | **88.44** | 82.73 | 84.45 | — | 81.88 |
| description | BLEU-4 | **48.26** | 44.64 | 45.56 | 40.26 | 46.14 |
| description | METEOR | **60.39** | 57.09 | 58.07 | — | 58.14 |

**Reading the IUPAC row.** This project retrieves names from PubChem instead of
generating them. MolScribe runs with three deterministic views (`base`, 8% pad,
+3° rotation); the image-only FG head picks the most consistent view, and the
base result is kept when it is already valid. If the selected view has no
PubChem record, the other views are tried before giving up. On the 90.56% of
test molecules where a record is found, BLEU-4 is **99.03** and **91.14%** of
names are character-identical to the ground truth; end-to-end, 82.5% of all
2,680 names are exact.

All four headline metrics (SMILES Acc_s, functional-group F1, IUPAC BLEU-4 and
description BLEU-4) meet or beat the best OCSR-based column. Functional groups
beat both Mol-VL sizes, IUPAC beats both OCSR-based columns, and SMILES Acc_s,
Acc_c and Tanimoto also exceed both paper OCSR columns.

---

## How the two paper routes compare with this project

| | **Paper: OCSR-based** | **Paper: OCSR-free** | **This project** |
|---|---|---|---|
| Core idea | `P(T\|I) = P(S\|I) · P(T\|S)`: recognise the structure, then understand it | a single VLM learns `P(T\|I)` end to end | OCSR-based, with an image-only functional-group head |
| Image → SMILES | DoubleCheck or MolScribe | Mol-VL generates SMILES directly | **MolScribe** public checkpoint, multi-view TTA |
| Functional groups | RDKit + OCSU dictionary | Mol-VL generates | **ConvNeXt-T image head** (default); dictionary fallback |
| IUPAC | PubChem retrieval | Mol-VL generates | **PubChem retrieval** |
| Description | BioT5+ | Mol-VL generates | **BioT5+**, SMILES canonicalised first |
| Name grounding | database record | none | **database record** |
| Training | OCSR trained separately (DoubleCheck 4×A100×4 days; MolScribe 4×A100×1 day); downstream modules not trained | 4×A100×10 days, 50 epochs, full-parameter SFT | **only the 27.8 M FG head**, about 30 minutes on an RTX 4060; OCSR and BioT5+ use public weights |
| Inference stack | OCSR + BioT5+ + PubChem | one 2B / 7B VLM | OCSR + FG head + BioT5+ + PubChem |
| Main strength | IUPAC names are real database records; modules replaceable | one model, strong generated text | real records + image FG head; small and reproducible |
| Main trade-off | names are PubChem records; OCSR is the only learned perception stage | one model for all tasks; names are generated rather than retrieved | OCSR + retrieval; modules can be replaced independently |

### Resource comparison

| item | **This project** | Paper: OCSR-based | Paper: OCSR-free |
|---|---|---|---|
| Parameters in the pipeline | **≈ 0.38 B** (MolScribe 95.5 M + ConvNeXt-T 27.8 M + BioT5+ 252 M) | OCSR (MolScribe-class, ≈ 95 M) + BioT5+ 252 M | 2 B / 8.29 B (7B) |
| Components that need training | **FG head only, 27.8 M** | OCSR backbone | full VLM |
| Training compute | **1× RTX 4060, ≈ 30 min** | DoubleCheck 4×A100×4 days; MolScribe 4×A100×1 day | 4×A100×10 days |
| Inference GPU | **RTX 4060 8 GB runs every module** | OCSR + text model | bf16 needs 24 GB+; 4-bit needs ≈ 6 GB but is slower |
| Disk for weights | **≈ 2.2 GB** | not stated | ≈ 16.6 GB for 7B |
| Full test, four tasks | **≈ 2.3 hours on one 4060** | not stated | ≈ 2.75 hours per task at 3 s/image (7B, 4-bit) |
| External dependency | PubChem network for IUPAC | PubChem network for IUPAC | none for generation |

The 8 GB claim is measured, not estimated: the full 3,300-image SMILES run,
3,269-image FG run, 2,680 PubChem lookups and 3,300 BioT5+ descriptions were all
executed on one RTX 4060.

---

## What it does

- MolScribe image → SMILES, vendored with compatibility patches; optional
  multi-view TTA selected by the image-only FG head.
- ConvNeXt-T 140-class functional-group head, trained from Vis-CheBI20 captions.
- PubChem IUPAC retrieval with canonical SMILES, InChIKey, Kekulé, CID and
  largest-fragment fallbacks; hit-only cache.
- BioT5+ description with RDKit canonicalisation before SELFIES.
- Optional OPSIN name → structure verification.
- Paper-aligned evaluation for all four tasks.

---

## Quick start

### 1. Install

Python 3.10+ and (optionally) Java 11+ for OPSIN verification.

```bash
git clone https://github.com/allforcloud/ocsu-hybrid.git
cd ocsu-hybrid

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install the CUDA build that matches your driver first.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

pip install -e ".[all]"
```

> This is a source-checkout project. Use `pip install -e .` from the repository:
> the MolScribe worker and the vendored MolScribe/OpenNMT source live in
> `tools/` and `third_party/`, outside the Python package. A plain wheel install
> is not the supported distribution.

CPU-only works too:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[all]"
```

### 2. Download the public weights

```bash
# MolScribe, BioT5+, OPSIN, NLTK data
python scripts/download_models.py --all

# Functional-group head (hosted on Hugging Face)
python scripts/download_models.py --fg \
  --fg-repo allforcloud9527/ocsu-hybrid-fg-convnext-t
```

Verify:

```bash
python scripts/download_models.py --check
```

Expected layout:

```
models/
├── molscribe/swin_base_char_aux_1m680k.pth
├── biot5-plus-chebi20/
├── fg_classifier/model.pt
└── opsin/opsin-cli.jar
```

### 3. Analyze one image

```bash
ocsu analyze examples/vischebi20_000000.png
```

Real output (description abbreviated):

```json
{
  "image": ".../examples/vischebi20_000000.png",
  "smiles": "[H][C@@]12CCC3=CC(=O)CC[C@]3(C)[C@@]1([H])C(=O)C[C@]1(C)/C(=C\\C(=O)OC)CC[C@@]21[H]",
  "canonical_smiles": "COC(=O)/C=C1/CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3C(=O)C[C@]12C",
  "valid": true,
  "functional_groups": [
    "alkenes", "carbonyl methyl ester", "cyclohexane", "cyclopentane",
    "esters", "ketones", "methoxy", "side-chain aldehydes"
  ],
  "iupac": {
    "name": "methyl (2Z)-2-[(8S,9S,10R,13S,14S)-10,13-dimethyl-3,11-dioxo-1,2,6,7,8,9,12,14,15,16-decahydrocyclopenta[a]phenanthren-17-ylidene]acetate",
    "source": "inchikey",
    "status": "ok"
  },
  "description": {
    "text": "The molecule is a 3-oxo Delta(4)-steroid that is ...",
    "status": "ok"
  }
}
```

Options:

```bash
# skip the slow stages
ocsu analyze molecule.png --no-iupac --no-description

# do not download the FG head; use the RDKit dictionary (91.11 F1)
ocsu analyze molecule.png --fg-backend dictionary

# verify the retrieved name with OPSIN
ocsu analyze molecule.png --verify-iupac

# force CPU
ocsu analyze molecule.png --device cpu
```

### 4. Batch

```bash
ocsu batch examples/ -o outputs/examples.jsonl
ocsu batch images/ -o outputs/images.jsonl --recursive
ocsu batch list.txt -o outputs/list.jsonl      # one path per line
```

Batch runs MolScribe once for the whole chunk, then batches the FG head and
parallelises PubChem lookups.

### 5. Run the full test

```bash
# Download Vis-CheBI20 from https://huggingface.co/datasets/PharMolix/Vis-CheBI20
# and point --data-dir at the directory containing test.json and imgs/.

ocsu eval --data-dir /path/to/vischebi20 --out results/full_test \
  --molscribe-views "base,pad:0.08,rot:3"
```

`--molscribe-views` enables multi-view OCSR test-time augmentation. The FG head
selects the most consistent view; the base result is kept when it is already
valid. Omit the flag for single-view inference (about 2.3× faster OCSR).

Smoke test first (20 samples per task):

```bash
ocsu eval --data-dir /path/to/vischebi20 --out results/smoke --limit 20
```

Equivalent script:

```bash
python scripts/run_full_test.py --data-dir /path/to/vischebi20 \
  --out results/full_test --iupac-workers 4 \
  --molscribe-views "base,pad:0.08,rot:3"
```

The runner resumes: OCSR, FG, IUPAC and description predictions are appended to
`results/full_test/preds/`, so an interrupted PubChem run does not redo the GPU
stages. Resume is keyed by image only; if you change the OCSR checkpoint, the
`--molscribe-views` setting or the FG backend, delete the corresponding
prediction file (or the whole `preds/` directory) to avoid mixing runs. Output:

```
results/full_test/
├── metrics.json
├── summary.txt
├── preds/{ocsr,smiles,fg,iupac,desc}.jsonl
└── details/{smiles,fg,iupac,desc}.jsonl
```

### 6. Python API

```python
from ocsu_hybrid.pipeline import Pipeline

pipeline = Pipeline()                       # defaults to all four stages
result = pipeline.analyze("examples/vischebi20_000000.png")
print(result["smiles"])
print(result["functional_groups"])
print(result["iupac"]["name"])
print(result["description"]["text"])

rows = pipeline.analyze_batch(["a.png", "b.png"], use_description=False)
```

---

## Reproducing the reported numbers

The metrics above were produced with the same evaluation code that ships in
`src/ocsu_hybrid/evaluate/`. To reproduce them on the full split:

```bash
ocsu eval --data-dir /path/to/vischebi20 --out results/full_test \
  --iupac-workers 4 --molscribe-views "base,pad:0.08,rot:3"
```

Expected result on an RTX 4060 8 GB:

```
SMILES      Acc_s 85.42  Acc_g 88.79  Acc_c 82.07  Tanimoto 94.28
FG          F1 99.07  P 99.31  R 99.00  exact 92.87
IUPAC       coverage 90.56%  BL-4 71.97  R-1 88.44  METEOR 88.37
Description BL-4 48.26  R-2 49.21  R-L 57.27  METEOR 60.39
```

The values were independently re-scored from the full-test prediction files with
this repository's evaluator and match the table exactly. See
[`docs/evaluation.md`](docs/evaluation.md) for the metric definitions, the
end-to-end vs conditional convention, the SciBERT tokenizer, the F1
precision/recall column swap in the paper, and the BioT5+ tokenizer artifact.

---

## Models and data

| component | source | size | license | distribution |
|---|---|---|---:|---|
| MolScribe checkpoint | [`yujieq/MolScribe`](https://huggingface.co/yujieq/MolScribe) | 1.1 GB | MIT | Hugging Face |
| BioT5+ | [`QizhiPei/biot5-plus-base-chebi20`](https://huggingface.co/QizhiPei/biot5-plus-base-chebi20) | 0.96 GB | MIT | Hugging Face |
| FG ConvNeXt-T head | [`allforcloud9527/ocsu-hybrid-fg-convnext-t`](https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t) | 107 MB | Apache-2.0 | Hugging Face |
| OPSIN 2.9.0 | [dan2097/opsin](https://github.com/dan2097/opsin) | 14 MB | MIT | download |
| SciBERT vocabulary | [allenai/scibert](https://github.com/allenai/scibert) | 223 KB | Apache-2.0 | bundled |
| MolScribe source | [thomas0809/MolScribe](https://github.com/thomas0809/MolScribe) | 14 MB | MIT | bundled (vendored) |
| OpenNMT-py 2.2.0 minimal | [OpenNMT/OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py) | 1 MB | MIT | bundled (vendored) |
| Vis-CheBI20 | [`PharMolix/Vis-CheBI20`](https://huggingface.co/datasets/PharMolix/Vis-CheBI20) | 3,300 test images | Apache-2.0 | download; one example bundled |

The functional-group checkpoint is hosted on Hugging Face at
[`allforcloud9527/ocsu-hybrid-fg-convnext-t`](https://huggingface.co/allforcloud9527/ocsu-hybrid-fg-convnext-t).
[`docs/fg_model_card.md`](docs/fg_model_card.md) is the model card, and
`scripts/train_fg_classifier.py` is the training script.

If the head is unavailable, every command still runs with
`--fg-backend dictionary`; the only difference is functional-group F1 (91.11
instead of 99.07).

### Documentation

| document | content |
|---|---|
| [`docs/architecture.md`](docs/architecture.md) | stage-by-stage design, data flow, failure behaviour, runtime |
| [`docs/evaluation.md`](docs/evaluation.md) | metric definitions, paper evaluator reproduction, coverage vs conditional |
| [`docs/fg_model_card.md`](docs/fg_model_card.md) | Hugging Face model card for the functional-group head |

### Train the FG head yourself

```bash
python scripts/build_fg_dataset.py --data-json /path/to/train.json
python scripts/train_fg_classifier.py \
  --train-json /path/to/vischebi20/train.json \
  --images-root /path/to/vischebi20 \
  --out models/fg_classifier
```

This reproduces the 99.07 F1 checkpoint: ConvNeXt-T, 384², 140 classes, batch 32,
12 epochs, about 30 minutes on an RTX 4060.

---

## Environment variables

All paths can be overridden without editing code:

| variable | default |
|---|---|
| `OCSU_HOME` | repository root |
| `OCSU_MODELS` | `<root>/models` |
| `OCSU_DATA` | `<root>/data/vischebi20` |
| `OCSU_CACHE` | `<root>/.cache` |
| `MOLSCRIBE_CKPT` | `<models>/molscribe/swin_base_char_aux_1m680k.pth` |
| `MOLSCRIBE_SRC` | `<root>/third_party/MolScribe` |
| `OCSU_FG_CKPT` | `<models>/fg_classifier/model.pt` |
| `BIOT5_PATH` | `<models>/biot5-plus-chebi20` |
| `OCSU_OPSIN_JAR` | `<models>/opsin/opsin-cli.jar` |
| `MOLSCRIBE_PY` | current Python interpreter |

The MolScribe worker runs in the current interpreter by default. If you keep the
OCSR stack in a separate environment, set `MOLSCRIBE_PY=/path/to/python` or pass
`--molscribe-python`.

---

## Scope and guarantees

1. **Every IUPAC name is a PubChem record.** The pipeline returns an empty name
   rather than a guess when no record is found.
2. **Multi-view OCSR is selected by the image head.** MolScribe runs three
   deterministic views; the ConvNeXt-T head keeps the view whose functional
   groups best match the image; the other views are tried if the selected one
   has no PubChem record. This raises coverage to 90.56% and hit-only BLEU-4 to
   99.03.
3. **Functional-group labels follow the 140-class OCSU dictionary.** The head
   learns image → dictionary group; it does not localise groups or generate
   free-form chemistry text.
4. **Descriptions use the public BioT5+ tokenizer.** Outputs are scored as
   generated, including its multi-digit tokenisation artifact.

---

## Citation

If you use this project, please cite the OCSU paper and the component papers:

```bibtex
@article{fan2025ocsu,
  title   = {OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific Discovery},
  author  = {Fan, Siqi and Xie, Yuguang and Cai, Bowen and Xie, Ailin and Liu, Gaochao and Qiao, Mu and Xing, Jie and Nie, Zaiqing},
  journal = {arXiv preprint arXiv:2501.15415},
  year    = {2025}
}

@article{qian2023molscribe,
  title   = {MolScribe: Robust Molecular Structure Recognition with Image-to-Graph Generation},
  author  = {Qian, Yujie and Guo, Jiang and Tu, Zhengkai and Li, Zhen and Coley, Connor W. and Barzilay, Regina},
  journal = {Journal of Chemical Information and Modeling},
  year    = {2023}
}

@article{pei2024biot5plus,
  title   = {BioT5+: Towards Generalized Biological Understanding with IUPAC Integration and Multi-task Tuning},
  author  = {Pei, Qizhi and Wu, Lijun and Gao, Kaiyuan and Liang, Xiaozhuan and Fang, Yin and Zhu, Jinhua and Xie, Shufang and Qin, Tao and Yan, Rui},
  journal = {arXiv preprint arXiv:2402.17810},
  year    = {2024}
}
```

## Acknowledgements

MolScribe (MIT), BioT5+ (MIT), OPSIN (MIT), RDKit (BSD-3-Clause), SciBERT
(Apache-2.0), OpenNMT-py (MIT), and the OCSU authors for the Vis-CheBI20 dataset
and functional-group dictionary (Apache-2.0). See [`NOTICE`](NOTICE) for the full
attribution and license text.

## License

Code: **Apache-2.0**. Third-party models, data and vendored source retain their
own licenses; see [`NOTICE`](NOTICE).
