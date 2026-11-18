"""Dataset-style IUPAC name normalisation.

The rules below encode the naming conventions used by the Vis-CheBi20 ground
truth.  They only rewrite a name when the result is another PubChem synonym for
the same InChIKey, so the structure is never changed by this module.
"""
from __future__ import annotations

import re
from typing import Iterable, Optional

from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

_HETEROARYL = {
    "pyridinyl": "pyridin",
    "pyrimidinyl": "pyrimidin",
    "pyrazinyl": "pyrazin",
    "pyridazinyl": "pyridazin",
    "pyrazolyl": "pyrazol",
    "imidazolyl": "imidazol",
    "triazolyl": "triazol",
    "tetrazolyl": "tetrazol",
    "thiazolyl": "thiazol",
    "oxazolyl": "oxazol",
    "isoxazolyl": "isoxazol",
    "thiophenyl": "thiophen",
    "furanyl": "furan",
    "pyrrolidinyl": "pyrrolidin",
    "piperidinyl": "piperidin",
    "piperazinyl": "piperazin",
    "morpholinyl": "morpholin",
}
_HETEROARYL_RE = re.compile(
    r"(-?)(\d+)-(" + "|".join(_HETEROARYL) + r")\b", re.I)

_IONS = (
    "sodium|potassium|lithium|magnesium|calcium|barium|zinc|cobalt|copper|"
    "iron|manganese|nickel|lead|mercury|silver|aluminium|aluminum|ammonium|"
    "caesium|cesium|rubidium|strontium"
)
_ION_PREFIX_RE = re.compile(r"^((?:" + _IONS + r")(?:\(\d+\+\))?)\s+", re.I)
_HALIDE_SUFFIX_RE = re.compile(r"\s+(chloride|bromide|iodide)$", re.I)
_ZINC_BIS_RE = re.compile(r"^zinc bis\((.*)\)$", re.I)
_ISOTOPE_FLUORO_RE = re.compile(r"\((\d+[A-Z]?)\)fluoro\b")
_CIS_TRANS_RE = re.compile(r"\b(cis|trans)-", re.I)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().rstrip(".")).lower()


def normalize_name(name: str) -> str:
    """Return the dataset-style variant of *name* (not yet synonym-gated)."""
    if not name:
        return name
    value = _HETEROARYL_RE.sub(
        lambda m: f"{_HETEROARYL[m.group(3).lower()]}-{m.group(2)}-yl", name)
    value = _CIS_TRANS_RE.sub("", value)

    match = _ZINC_BIS_RE.match(value)
    if match:
        value = "zinc;" + match.group(1)

    match = _ION_PREFIX_RE.match(value)
    if match:
        value = value[:match.start()] + match.group(1) + ";" + value[match.end():]
    elif re.search(r"ium|" + _IONS, value, re.I):
        match = _HALIDE_SUFFIX_RE.search(value)
        if match:
            value = value[:match.start()] + ";" + match.group(1)

    return _ISOTOPE_FLUORO_RE.sub(r"(\1)fluoranyl", value)


def inchikey(smiles: Optional[str]) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def gated_name(name: str, synonyms: Iterable[str]) -> str:
    """Return the normalised name only when it is a known PubChem synonym."""
    candidate = normalize_name(name)
    if candidate == name:
        return name
    allowed = {normalize_text(item) for item in synonyms}
    return candidate if normalize_text(candidate) in allowed else name
