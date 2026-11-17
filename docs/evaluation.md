# Evaluation protocol

All numbers in the README use the full Vis-CheBI20 test split and the OCSU
paper's metric definitions. This document records the exact conventions so the
results can be reproduced and compared.

## Data

| task | `task_name` | samples |
|---|---|---:|
| SMILES | `trans_smiles` | 3,300 |
| functional groups | `struct_cap` | 3,269 |
| IUPAC | `trans_iupac` | 2,680 |
| description | `general_desp` | 3,300 |

Download from <https://huggingface.co/datasets/PharMolix/Vis-CheBI20> (Apache-2.0)
and point `ocsu eval --data-dir` at the directory containing `test.json` and the
`imgs/` tree.

## SMILES

Following `OCSU/scripts/evaluator.py::SmilesEvaluator`:

- `Acc_s`: canonical exact match after removing `/` and `\` (cis/trans ignored).
- `Acc_g`: canonical exact match after removing stereochemistry (graph match).
- `Acc_c`: `Acc_s` restricted to ground-truth molecules whose canonical SMILES
  contains `@` (1,969 samples).
- `Tanimoto`: mean RDKit fingerprint similarity.

R-group tokens such as `[R1]` are rewritten to `[1*]` before canonicalisation,
matching the paper's evaluator.

## Functional groups

`parse_fg()` is the paper's string parser. The score is the mean per-molecule F1
over predicted and ground-truth sets. We also report precision, recall and exact
set match.

**Column-name caveat.** The paper's `F1Evaluator` stores `inter/|GT|` under
"precision" and `inter/|pred|` under "recall". F1 is unaffected, but the last two
columns of the paper's Table 4 are, in standard terms, recall and precision. The
values in this project use the standard definitions.

## IUPAC and description

`CommonEvaluator` tokenises with `scibert_scivocab_uncased`, restricts to 512
tokens, and computes:

- corpus BLEU-2 and BLEU-4 (NLTK `corpus_bleu`)
- mean ROUGE-1 / ROUGE-2 / ROUGE-L F-measure (`rouge_score`)
- mean METEOR (`nltk.translate.meteor_score`)

This repository vendors only the 223 KB SciBERT vocabulary
(`src/ocsu_hybrid/assets/scibert/vocab.txt`, Apache-2.0), so the tokenisation is
identical without downloading the 842 MB model. The first run downloads the NLTK
WordNet corpora; `python scripts/download_models.py --nltk` does this explicitly.

### End-to-end vs conditional

- **end-to-end** scores every test sample and counts a missing prediction as an
  empty string. This is the number comparable to the paper's single-pass table.
- **conditional** scores only samples with a prediction. It measures the
  retrieval/generation stage in isolation.

For IUPAC, the gap between the two is almost entirely coverage:

| quantity | value |
|---|---:|
| coverage | 87.72% (2,351 / 2,680) |
| BLEU-4 on covered samples | 99.01 |
| exact name on covered samples | 91.20% |
| exact name end-to-end | 80.0% (2,144 / 2,680) |

The 329 misses have a median predicted-vs-ground-truth Tanimoto of 0.794: they
are mostly OCSR errors, not PubChem lookup failures. Querying the ground-truth
SMILES finds a PubChem record for 7 of 10 sampled misses, which confirms the
lookup layer is not the main gap.

### OPSIN

OPSIN is a verification gate, not a name source. `--verify-iupac` converts each
retrieved name back to a structure and compares it with the SMILES it came from.
The result is one of `verified` (exact or stereo-insensitive), `mismatch`,
`unparsable`, or `unavailable` when Java or the jar is missing.

### BERTScore

BERTScore is optional (`--with-bertscore`) and not part of the main table. The
paper does not state whether it uses `rescale_with_baseline`; both raw and
rescaled values are reported when enabled. Without the authors' configuration,
BS-F1 is the one metric that is not directly comparable.

### BioT5+ tokenizer artifact

The public BioT5+ tokenizer splits multi-digit numbers, so `17-oxo` decodes as
`1 7 -oxo` and `Delta(4)` as `Delta( 4 )`. This is a property of the released
checkpoint and applies equally to the paper's BioT5+ baseline. Outputs are scored
as generated, without post-hoc cleanup.

## Reproduce

```bash
ocsu eval --data-dir /path/to/vischebi20 --out results/full_test
```

The runner writes:

```
results/full_test/
├── metrics.json          # all task metrics
├── summary.txt           # one-screen summary
├── preds/
│   ├── ocsr.jsonl        # image, smiles, raw, latency
│   ├── smiles.jsonl      # image, pred
│   ├── fg.jsonl          # image, pred[]
│   ├── iupac.jsonl       # image, pred, source
│   └── desc.jsonl        # image, pred
└── details/              # per-sample scores for all four tasks
```

A 20-image smoke test takes a few minutes; the full run takes about two hours on
an RTX 4060. `--limit 20` caps each task, not the whole file.

## Reported values

| task | metric | value |
|---|---|---:|
| SMILES | Acc_s | 84.00 |
| SMILES | Acc_g | 86.94 |
| SMILES | Acc_c | 80.55 |
| SMILES | Tanimoto | 92.03 |
| FG | F1 | 99.07 |
| FG | precision / recall | 99.31 / 99.00 |
| FG | exact set | 92.87 |
| IUPAC | coverage | 87.72 |
| IUPAC | BLEU-4 end-to-end | 65.98 |
| IUPAC | BLEU-4 conditional | 99.01 |
| IUPAC | ROUGE-1 end-to-end | 85.39 |
| IUPAC | METEOR end-to-end | 85.41 |
| description | BLEU-4 | 46.83 |
| description | ROUGE-2 | 47.96 |
| description | ROUGE-L | 55.89 |
| description | METEOR | 58.99 |

These values were reproduced from the full-test prediction files with the
`src/ocsu_hybrid/evaluate` code in this repository.
