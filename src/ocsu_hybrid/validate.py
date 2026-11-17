"""SMILES validation and canonicalisation."""
from __future__ import annotations

from typing import Any, Dict, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")


def validate_smiles(smiles: Optional[str]) -> Dict[str, Any]:
    """Return a small, explicit validation record. No fallback, no guessing."""
    if not smiles or not smiles.strip():
        return dict(valid=False, canonical=None, heavy_atoms=0, fragments=0,
                    error="empty_smiles")
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return dict(valid=False, canonical=None, heavy_atoms=0, fragments=0,
                    error="rdkit_parse_failed")
    return dict(
        valid=True,
        canonical=Chem.MolToSmiles(mol, isomericSmiles=True),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        fragments=len(Chem.GetMolFrags(mol)),
        error=None,
    )
