"""IUPAC names via PubChem, matching the OCSU OCSR-based route.

Every returned name is a PubChem record; nothing is generated. The lookup chain
handles the query forms that PubChem rejects or normalises differently:

1. canonical SMILES
2. InChIKey (stable across SMILES writing styles)
3. Kekule SMILES
4. CID lookup
5. largest fragment (salts / mixtures)

Hits are cached as JSONL. Misses are never cached: a transient rate limit must
not turn into a permanent "not in PubChem" verdict.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem, RDLogger

from .. import paths

RDLogger.DisableLog("rdApp.*")

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"


class PubChemResolver:
    def __init__(self, cache_path: str | Path | None = None, *,
                 timeout: int = 15, retries: int = 3, sleep_s: float = 0.2,
                 user_agent: str = "ocsu-hybrid/0.1"):
        self.cache_path = Path(cache_path) if cache_path else (
            paths.cache_dir() / "iupac_pubchem.jsonl")
        self.timeout = timeout
        self.retries = retries
        self.sleep_s = sleep_s
        self.user_agent = user_agent
        self._cache: Optional[dict] = None
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ cache
    def _load_cache(self) -> dict:
        with self._lock:
            if self._cache is None:
                self._cache = {}
                if self.cache_path.exists():
                    for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                        try:
                            row = json.loads(line)
                            self._cache[row["smiles"]] = (row.get("name"),
                                                          row.get("source"))
                        except Exception:  # noqa: S112 - skip malformed cache lines
                            continue
            return self._cache

    def _save_cache(self, smiles: str, name: str, source: str) -> None:
        with self._lock:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(dict(smiles=smiles, name=name, source=source),
                                   ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------- http
    def _get_json(self, url: str):
        for attempt in range(self.retries):
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": self.user_agent})
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode())
            except urllib.error.HTTPError as exc:
                # 4xx means "no such compound"; retrying cannot help.
                if 400 <= exc.code < 500:
                    return None
                if attempt + 1 < self.retries:
                    time.sleep(0.6 * (attempt + 1))
            except Exception:
                if attempt + 1 < self.retries:
                    time.sleep(0.6 * (attempt + 1))
        return None

    def _iupac_from_props(self, section: str, key: str) -> Optional[str]:
        url = (f"{PUG}/compound/{section}/{urllib.parse.quote(key, safe='')}"
               f"/property/IUPACName/JSON")
        data = self._get_json(url)
        try:
            return data["PropertyTable"]["Properties"][0]["IUPACName"]
        except Exception:
            return None

    def _cid_from_smiles(self, smiles: str) -> Optional[str]:
        url = f"{PUG}/compound/smiles/{urllib.parse.quote(smiles, safe='')}/cids/JSON"
        data = self._get_json(url)
        try:
            return str(data["IdentifierList"]["CID"][0])
        except Exception:
            return None

    # ------------------------------------------------------------------- chem
    @staticmethod
    def _canonical(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None

    @staticmethod
    def _kekule(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            return Chem.MolToSmiles(mol, isomericSmiles=True, kekuleSmiles=True)
        except Exception:
            return None

    @staticmethod
    def _inchikey(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            return Chem.MolToInchiKey(mol)
        except Exception:
            return None

    @staticmethod
    def _largest_fragment(smiles: str) -> Optional[str]:
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        if mol is None:
            return None
        try:
            frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        except Exception:
            return None
        if len(frags) <= 1:
            return None
        best = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return Chem.MolToSmiles(best, isomericSmiles=True)

    # -------------------------------------------------------------------- api
    def resolve(self, smiles: str) -> Tuple[Optional[str], Optional[str]]:
        """Return ``(name, source)``.

        ``source`` is one of ``smiles``, ``inchikey``, ``smiles_kekule``, ``cid``,
        ``fragment`` or ``cache``.
        """
        if not smiles:
            return None, None
        canonical = self._canonical(smiles)
        if not canonical:
            return None, None

        cache = self._load_cache()
        with self._lock:
            if canonical in cache:
                name, source = cache[canonical]
                # A cached miss may come from an older/rate-limited run; retry it
                # instead of treating it as a permanent verdict.
                if name:
                    return name, source or "cache"

        name, source = None, None

        name = self._iupac_from_props("smiles", canonical)
        if name:
            source = "smiles"

        if not name:
            key = self._inchikey(canonical)
            if key:
                name = self._iupac_from_props("inchikey", key)
                if name:
                    source = "inchikey"

        if not name:
            kekule = self._kekule(canonical)
            if kekule and kekule != canonical:
                name = self._iupac_from_props("smiles", kekule)
                if name:
                    source = "smiles_kekule"

        if not name:
            kekule = self._kekule(canonical) or canonical
            cid = self._cid_from_smiles(canonical) or self._cid_from_smiles(kekule)
            if cid:
                name = self._iupac_from_props("cid", cid)
                if name:
                    source = "cid"

        if not name:
            fragment = self._largest_fragment(canonical)
            if fragment:
                key = self._inchikey(fragment)
                if key:
                    name = self._iupac_from_props("inchikey", key)
                    if name:
                        source = "fragment"

        if name:
            with self._lock:
                cache[canonical] = (name, source)
            self._save_cache(canonical, name, source)
        time.sleep(self.sleep_s)
        return name, source

    def resolve_many(self, smiles: Sequence[Optional[str]], workers: int = 4
                     ) -> List[Tuple[Optional[str], Optional[str]]]:
        """Resolve a list concurrently, preserving order and de-duplicating queries."""
        unique = []
        seen = set()
        for value in smiles:
            if value and value not in seen:
                seen.add(value)
                unique.append(value)

        results = {}
        if unique:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                for value, result in zip(unique, pool.map(self.resolve, unique)):
                    results[value] = result
        return [results.get(value, (None, None)) if value else (None, None)
                for value in smiles]


_default_resolver: Optional[PubChemResolver] = None
_default_lock = threading.Lock()


def default_resolver() -> PubChemResolver:
    global _default_resolver
    with _default_lock:
        if _default_resolver is None:
            _default_resolver = PubChemResolver()
        return _default_resolver


def iupac_name(smiles: str, *, cache_path: str | Path | None = None
               ) -> Tuple[Optional[str], Optional[str]]:
    """Convenience wrapper around :class:`PubChemResolver`."""
    if cache_path is None:
        return default_resolver().resolve(smiles)
    return PubChemResolver(cache_path).resolve(smiles)


def resolve_many(smiles: Iterable[Optional[str]], workers: int = 4
                 ) -> List[Tuple[Optional[str], Optional[str]]]:
    return default_resolver().resolve_many(list(smiles), workers=workers)
