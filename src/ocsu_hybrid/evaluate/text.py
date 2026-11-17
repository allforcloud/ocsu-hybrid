"""Text metrics, byte-compatible with the OCSU ``CommonEvaluator``.

The paper tokenises with ``scibert_scivocab_uncased`` and then computes corpus
BLEU-2/4, mean ROUGE-1/2/L and mean METEOR. Only the tokenizer vocabulary is
needed, so this project vendors the 223 KB ``vocab.txt`` (SciBERT, Apache-2.0)
instead of the 842 MB model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .. import paths


class TextScorer:
    def __init__(self, vocab_file: str | Path | None = None):
        from nltk.translate.meteor_score import meteor_score  # noqa: F401
        from rouge_score import rouge_scorer
        from transformers import BertTokenizerFast

        vocab = Path(vocab_file) if vocab_file else paths.scibert_vocab()
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"])
        self._meteor = meteor_score
        self._ensure_nltk()

    @staticmethod
    def _ensure_nltk() -> None:
        import nltk

        for resource in ("wordnet", "omw-1.4"):
            try:
                nltk.data.find(f"corpora/{resource}")
            except LookupError:
                try:
                    nltk.download(resource, quiet=True)
                except Exception as exc:  # noqa: BLE001
                    raise RuntimeError(
                        f"NLTK resource '{resource}' is required for METEOR. "
                        "Run `python scripts/download_models.py --nltk` once."
                    ) from exc

    def _tokenize(self, text: str, max_length: int = 512) -> List[str]:
        try:
            tokens = self.tokenizer.tokenize(text or "", truncation=True,
                                             max_length=max_length,
                                             padding="max_length")
        except TypeError:
            tokens = self.tokenizer.tokenize(text or "")[:max_length]
        return [t for t in tokens if t not in ("[PAD]", "[CLS]", "[SEP]")]

    def evaluate(self, pairs: Sequence[dict]) -> Dict[str, float]:
        """``pairs`` is a sequence of ``{"gt": str, "pred": str}``."""
        from nltk.translate.bleu_score import corpus_bleu

        if not pairs:
            return {"Num Samples:": 0.0, "BLEU-2": 0.0, "BLEU-4": 0.0,
                    "ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0,
                    "METEOR": 0.0}
        gt_tokens, pred_tokens, meteor, rouge = [], [], [], []
        for pair in pairs:
            gold = self._tokenize(pair["gt"])
            pred = self._tokenize(pair.get("pred") or "")
            gt_tokens.append([gold])
            pred_tokens.append(pred)
            meteor.append(self._meteor([gold], pred))
            rouge.append(self.scorer.score(pair.get("pred") or "", pair["gt"]))

        return {
            "Num Samples:": float(len(pairs)),
            "BLEU-2": float(corpus_bleu(gt_tokens, pred_tokens, weights=(0.5, 0.5))),
            "BLEU-4": float(corpus_bleu(gt_tokens, pred_tokens,
                                         weights=(0.25, 0.25, 0.25, 0.25))),
            "ROUGE-1": float(np.mean([r["rouge1"].fmeasure for r in rouge]))
            if rouge else 0.0,
            "ROUGE-2": float(np.mean([r["rouge2"].fmeasure for r in rouge]))
            if rouge else 0.0,
            "ROUGE-L": float(np.mean([r["rougeL"].fmeasure for r in rouge]))
            if rouge else 0.0,
            "METEOR": float(np.mean(meteor)) if meteor else 0.0,
        }


def bertscore(pairs: Sequence[dict], model_type: str = "roberta-large",
              rescale: bool = False, device: Optional[str] = None) -> float:
    import bert_score as bs_pkg
    from bert_score import score

    kwargs = dict(lang="en", verbose=False, batch_size=16,
                  device=device or ("cuda" if _cuda() else "cpu"))
    if model_type.startswith(("/", ".")):
        kwargs["num_layers"] = 24
    if rescale:
        kwargs["rescale_with_baseline"] = True
        kwargs["baseline_path"] = str(
            Path(bs_pkg.__file__).parent / "rescale_baseline" / "en" / "roberta-large.tsv")
    _, _, f1 = score([p.get("pred") or "" for p in pairs],
                     [p["gt"] for p in pairs], model_type=model_type, **kwargs)
    return float(f1.mean())


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False
