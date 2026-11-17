"""Functional groups from the OCSU SMARTS/SMILES dictionary.

This is the deterministic fallback backend. It sees only SMILES connectivity,
so its ceiling is set by OCSR graph accuracy (91.11 F1 with MolScribe on the
full Vis-CheBI20 test split).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_DATA = Path(__file__).with_name("fg_data.json")


@lru_cache(maxsize=1)
def _patterns() -> Tuple[List[Tuple[str, object]], List[str]]:
    data = json.loads(_DATA.read_text(encoding="utf-8"))
    compiled, broken = [], []
    for name, smarts in data.get("smart", {}).items():
        mol = Chem.MolFromSmarts(smarts)
        if mol is None:
            broken.append(f"SMARTS:{name}")
        else:
            compiled.append((name, mol))
    for name, smi in data.get("smiles", {}).items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            broken.append(f"SMILES:{name}")
        else:
            compiled.append((name, mol))
    return compiled, broken


def get_func_groups(smiles: Optional[str]) -> List[str]:
    """Names of all dictionary groups found in ``smiles`` (deduplicated)."""
    if not smiles:
        return []
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return []
    found, seen = [], set()
    for name, pattern in _patterns()[0]:
        if name in seen:
            continue
        if mol.HasSubstructMatch(pattern, useChirality=False):
            seen.add(name)
            found.append(name)
    return sorted(found)


def dictionary_stats() -> dict:
    compiled, broken = _patterns()
    return dict(compiled=len(compiled), broken=broken)
