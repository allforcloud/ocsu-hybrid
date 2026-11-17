"""Description via BioT5+ (SMILES -> SELFIES -> text).

SMILES is RDKit-canonicalised before SELFIES encoding. MolScribe emits explicit
hydrogens and unusual ring closures; SELFIES built from those is out of
distribution for BioT5+ (description BLEU-4 34 vs 51 on the explicit-H subset),
while canonicalisation is chemically neutral.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional

from .. import paths

TASK = ("Definition: You are given a molecule SELFIES. Your job is to generate "
        "the molecule description in English that fits the molecule SELFIES.\n\n")


class BioT5Describer:
    def __init__(self, model_path: str | Path | None = None,
                 device: str | None = None):
        import torch
        from transformers import T5ForConditionalGeneration, T5Tokenizer

        path = Path(model_path) if model_path else paths.biot5_path()
        paths.require(
            path,
            "Download BioT5+ with `python scripts/download_models.py --biot5` "
            "or pass --biot5-path / set BIOT5_PATH.",
        )
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.tok = T5Tokenizer.from_pretrained(str(path), model_max_length=512)
        self.model = T5ForConditionalGeneration.from_pretrained(
            str(path), torch_dtype=dtype, device_map=self.device).eval()

    def describe(self, smiles: str, max_length: int = 512) -> Optional[str]:
        if not smiles:
            return None
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        canonical = Chem.MolToSmiles(mol)
        try:
            import selfies

            sel = selfies.encoder(canonical)
        except Exception:
            return None
        prompt = TASK + f"Now complete the following example -\nInput: <bom>{sel}<eom>\nOutput: "
        ids = self.tok(prompt, return_tensors="pt", truncation=True,
                       max_length=512).input_ids.to(self.model.device)
        with self.torch.no_grad():
            out = self.model.generate(ids, max_length=max_length, num_beams=1)
        return self.tok.decode(out[0], skip_special_tokens=True).strip()

    def describe_many(self, smiles: Iterable[Optional[str]],
                      max_length: int = 512) -> List[Optional[str]]:
        out = []
        for value in smiles:
            if not value:
                out.append(None)
                continue
            try:
                out.append(self.describe(value, max_length=max_length))
            except Exception:  # noqa: BLE001 - one bad molecule must not stop a run
                out.append(None)
        return out
