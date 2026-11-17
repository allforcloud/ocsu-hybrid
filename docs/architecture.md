# Architecture

OCSU-Hybrid is a modular implementation of the OCSR-based route in
*OCSU: Optical Chemical Structure Understanding for Molecule-centric Scientific
Discovery* (arXiv 2501.15415). A molecule image is converted to SMILES first;
the other three tasks consume that SMILES or the image directly.

```
                         ┌──────────────────────────────────────────┐
 image ──► MolScribe ──► │ SMILES ──► PubChem ──► IUPAC name        │
           (95.5 M)      │    │                                     │
                         │    ├────► BioT5+ ──► description         │
                         │    │      (252 M)                        │
                         │    └────► RDKit dictionary ──► FG        │
                         │            (fallback backend)           │
                         └──────────────────────────────────────────┘
                              image ──► ConvNeXt-T ──► FG
                                        (27.8 M, default)
```

## Stages

| stage | implementation | output | notes |
|---|---|---|---|
| OCSR | MolScribe `swin_base_char_aux_1m680k` | SMILES | public checkpoint; persistent worker |
| functional groups | ConvNeXt-T multi-label head | up to 140 names | image-only; `--fg-backend dictionary` switches to RDKit |
| IUPAC | PubChem PUG REST | name or miss | no generation; hits cached |
| description | BioT5+ `biot5-plus-base-chebi20` | English text | SMILES canonicalised before SELFIES |

The default pipeline never loads a 7B model. The four stages together are about
0.38 B parameters and fit in 8 GB of GPU memory.

## Why these choices

**MolScribe instead of DoubleCheck.** DoubleCheck is the paper's best OCSR
backbone, but its weights are not public. MolScribe is the paper's other OCSR
baseline, is MIT-licensed, and reaches 84.00 Acc_s on the full test split
(paper: 80.50). The adapter is a single module, so a different OCSR checkpoint
can be dropped in without touching the downstream stages.

**Learned functional-group head instead of the dictionary.** The deterministic
`RDKit + OCSU dictionary` path can only see SMILES connectivity; its ceiling is
set by OCSR graph accuracy (91.11 F1). The ConvNeXt-T head reads the image
directly and reaches 99.07 F1. The dictionary remains available because it needs
no FG checkpoint and is useful for debugging.

**PubChem retrieval instead of name generation.** Every returned IUPAC name is a
database record. On the 87.72% of test molecules where the lookup succeeds,
BLEU-4 is 99.01 and 91.20% of names are character-identical to the ground truth.
A generative fallback can raise BLEU by producing plausible n-grams without
adding structurally correct records, so the shipped pipeline returns only
PubChem records and leaves misses empty.

**BioT5+ with RDKit canonicalisation.** MolScribe occasionally emits explicit
hydrogens and unusual ring closures; SELFIES built from those is out of
distribution for BioT5+. Canonicalising with RDKit is chemically neutral and
lifts description BLEU-4 from 44.85 to 46.83.

## Request flow

1. `Pipeline.analyze()` calls `Pipeline.analyze_batch()` with one image.
2. Images are resolved to absolute paths.
3. `MolScribeOCR` spawns `tools/molscribe_worker.py` once, sends a JSON list of
   images, and reads a JSONL result file. The worker loads the Swin encoder and
   graph decoder, preprocesses with OpenCV (crop white border → 384 px → grayscale
   → ImageNet normalisation) and writes SMILES.
4. RDKit validates each SMILES and produces a canonical form.
5. Functional groups come from the ConvNeXt-T head (batched) or the dictionary.
6. PubChem lookups run concurrently (`--iupac-workers`, default 4) and only hits
   are cached.
7. BioT5+ generates one description per valid canonical SMILES.
8. Optional OPSIN verification checks a retrieved name against the structure.

## Failure behaviour

| failure | behaviour |
|---|---|
| MolScribe returns invalid SMILES | `valid=false`; IUPAC/description are marked invalid, FG still runs (image-only) |
| PubChem has no record | `iupac.status="miss"`; the record stays empty (never guessed) |
| BioT5+ SELFIES encoding fails | `description.status="failed"` |
| OPSIN jar missing | `verification.status="unavailable"` (not a silent pass) |
| FG checkpoint missing | actionable `FileNotFoundError` with the download command |

## Repository layout

```
src/ocsu_hybrid/
  pipeline.py            orchestration
  cli.py                 ocsu analyze / batch / eval
  credentials.py         local-only publishing credentials (gitignored)
  full_test.py           full Vis-CheBI20 runner
  ocsr/molscribe.py      worker adapter
  fg/classifier.py       ConvNeXt-T head
  fg/dictionary.py       RDKit + OCSU dictionary
  iupac/pubchem.py       retrieval + cache
  iupac/opsin.py         name → structure gate
  describe/biot5.py      SMILES → SELFIES → text
  evaluate/              paper-aligned metrics
tools/molscribe_worker.py
third_party/MolScribe/   vendored, MIT, compatibility patches
third_party/onmt/        minimal OpenNMT-py 2.2.0 package, MIT
```

## Runtime

Measured on an RTX 4060 8 GB with the full test split:

| stage | time |
|---|---|
| MolScribe, 3,300 images | about 8 minutes at batch 8 |
| ConvNeXt-T FG, 3,269 images | under 1 minute |
| PubChem, 2,680 lookups at 4 workers | about 45 minutes |
| BioT5+, 3,300 descriptions | about 40 minutes |
| scoring | about 15 minutes |
| **total** | **about 2 hours** |

Single-image latency after model load is about 0.2–0.5 s for MolScribe and a few
seconds for a BioT5+ description; the first call pays the model-loading cost.
