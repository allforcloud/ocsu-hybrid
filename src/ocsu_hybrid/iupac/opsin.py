"""IUPAC -> structure verification with OPSIN.

OPSIN is a gate, never a fallback: the pipeline only uses it to check whether a
name really describes the structure it came from. An unavailable jar degrades to
``status="unavailable"`` rather than inventing a result.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from rdkit import Chem, RDLogger

from .. import paths

RDLogger.DisableLog("rdApp.*")


def _canon(smiles: str, stereo: bool = True) -> Optional[str]:
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=stereo)


def verify(name: str, smiles: str, *, jar: Optional[str | Path] = None,
           timeout: int = 30) -> Dict[str, Any]:
    """Return verified/mismatch/unparsable; never substitute another source."""
    path = Path(jar) if jar else paths.opsin_jar()
    name = re.sub(r"^(the\s+)?iupac\s*(is|:)\s*", "", (name or "").strip(), flags=re.I)
    name = name.replace("**", "").replace("`", "").replace(";", " ")
    name = name.strip().rstrip(".").strip()
    if not name or not smiles:
        return dict(status="unparsable", reason="empty_input")
    if not path.exists():
        return dict(status="unavailable", reason=f"missing_jar:{path}")
    try:
        proc = subprocess.run(
            ["java", "-jar", str(path)],
            input=name + "\n", capture_output=True, text=True, timeout=timeout,
            check=False,
        )
    except Exception as exc:  # noqa: BLE001
        return dict(status="error", reason=f"{type(exc).__name__}: {exc}")
    lines = [x.strip() for x in (proc.stdout or "").splitlines() if x.strip()]
    lines = [x for x in lines if not x.startswith("Run the jar")]
    if not lines:
        return dict(status="unparsable", reason="empty_opsin_output")
    line = lines[0]
    low = line.lower()
    if "unparsable" in low or "not parseable" in low or "not interpretable" in low:
        return dict(status="unparsable", reason=line[:200])
    got = _canon(line)
    want = _canon(smiles)
    if got is None or want is None:
        return dict(status="unparsable", reason="rdkit_parse_failed", structure=line)
    if got == want:
        return dict(status="verified", match="exact", structure=got)
    if _canon(line, stereo=False) == _canon(smiles, stereo=False):
        return dict(status="verified", match="no_stereo", structure=got)
    return dict(status="mismatch", match="different", structure=got, expected=want)
