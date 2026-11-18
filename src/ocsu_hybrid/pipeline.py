"""End-to-end OCSU-Hybrid pipeline.

Stages are independent and lazily loaded:

    MolScribe -> SMILES
    ConvNeXt-T (or RDKit dictionary) -> functional groups
    PubChem -> IUPAC
    BioT5+ -> description
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .describe.biot5 import BioT5Describer
from .fg.classifier import FGClassifier
from .fg.dictionary import get_func_groups
from .iupac.opsin import verify as verify_name
from .iupac.pubchem import PubChemResolver
from .ocsr.molscribe import MolScribeOCR
from .validate import validate_smiles


class Pipeline:
    def __init__(self, *, molscribe_ckpt: str | Path | None = None,
                 molscribe_python: str | None = None,
                 molscribe_batch_size: int = 8,
                 molscribe_views: str = "base",
                 fg_backend: str = "classifier",
                 fg_ckpt: str | Path | None = None,
                 device: str | None = None,
                 iupac_workers: int = 4,
                 iupac_cache: str | Path | None = None,
                 biot5_path: str | Path | None = None,
                 opsin_jar: str | Path | None = None,
                 use_iupac: bool = True,
                 use_description: bool = True,
                 verify_iupac: bool = False,
                 fg_batch_size: int = 32):
        if fg_backend not in ("classifier", "dictionary"):
            raise ValueError("fg_backend must be 'classifier' or 'dictionary'")
        self.fg_backend = fg_backend
        self.device = device
        self.use_iupac = use_iupac
        self.use_description = use_description
        self.verify_iupac = verify_iupac
        self.fg_batch_size = fg_batch_size
        self.iupac_workers = iupac_workers

        self._ocr = MolScribeOCR(ckpt=molscribe_ckpt, python=molscribe_python,
                                 batch_size=molscribe_batch_size,
                                 views=molscribe_views,
                                 dump_candidates=molscribe_views != "base")
        self._fg_ckpt = fg_ckpt
        self._fg: Optional[FGClassifier] = None
        self._resolver: Optional[PubChemResolver] = None
        self._iupac_cache = iupac_cache
        self._describer: Optional[BioT5Describer] = None
        self._biot5_path = biot5_path
        self._opsin_jar = opsin_jar

    # ------------------------------------------------------------- lazy stages
    def fg(self) -> FGClassifier:
        if self.fg_backend == "dictionary":
            raise RuntimeError("the dictionary backend does not load a classifier")
        if self._fg is None:
            self._fg = FGClassifier(self._fg_ckpt, device=self.device)
        return self._fg

    def resolver(self) -> PubChemResolver:
        if self._resolver is None:
            self._resolver = PubChemResolver(self._iupac_cache)
        return self._resolver

    def describer(self) -> BioT5Describer:
        if self._describer is None:
            self._describer = BioT5Describer(self._biot5_path, device=self.device)
        return self._describer

    # ------------------------------------------------------------------ stages
    def recognize(self, images: Sequence[str | Path]) -> List[dict]:
        return self._ocr.recognize_batch(images)

    @staticmethod
    def _fg_f1(left: Sequence[str], right: Sequence[str]) -> float:
        a, b = set(left), set(right)
        if not a or not b:
            return 0.0
        inter = len(a & b)
        precision = inter / len(b)
        recall = inter / len(a)
        return (2 * precision * recall / (precision + recall)
                if precision + recall else 0.0)

    @staticmethod
    def select_candidates(rows: List[dict],
                          image_groups: Sequence[Sequence[str]]) -> List[dict]:
        """Pick the TTA view whose dictionary FG set best matches the image head.

        The image FG head is independent of OCSR, so it is a useful tie-breaker
        between views. If a row has no valid candidate, keep its base result.
        """
        for row, groups in zip(rows, image_groups):
            candidates = row.get("candidates") or []
            if not candidates:
                continue
            best = None
            best_score = None
            for candidate in candidates:
                smiles = candidate.get("smiles")
                if not smiles:
                    continue
                score = (Pipeline._fg_f1(get_func_groups(smiles), groups),
                         candidate.get("confidence") or 0.0)
                if best_score is None or score > best_score:
                    best_score = score
                    best = candidate
            if best is not None:
                row["smiles"] = best["smiles"]
                row["raw"] = best.get("raw", row.get("raw"))
                row["selected_view"] = best.get("view")
                row["selected_confidence"] = best.get("confidence")
            elif row.get("smiles") is None:
                base = next((c for c in candidates if c.get("view") == "base"), None)
                if base:
                    row["smiles"] = base["smiles"]
        return rows

    def analyze(self, image: str | Path, **overrides) -> dict:
        return self.analyze_batch([image], **overrides)[0]

    def analyze_batch(self, images: Iterable[str | Path], *,
                      use_iupac: Optional[bool] = None,
                      use_description: Optional[bool] = None,
                      verify_iupac: Optional[bool] = None) -> List[dict]:
        paths = [Path(image).resolve() for image in images]
        if not paths:
            return []
        use_iupac = self.use_iupac if use_iupac is None else use_iupac
        use_description = (self.use_description if use_description is None
                           else use_description)
        check_iupac = self.verify_iupac if verify_iupac is None else verify_iupac

        started = time.perf_counter()
        ocr_rows = self.recognize(paths)
        ocr_time = time.perf_counter() - started

        # Functional groups -------------------------------------------------
        fg_started = time.perf_counter()
        if self.fg_backend == "classifier":
            groups = self.fg().predict(paths, batch_size=self.fg_batch_size)
        else:
            groups = [get_func_groups(row.get("smiles")) if row.get("smiles") else []
                      for row in ocr_rows]
        fg_time = time.perf_counter() - fg_started

        if getattr(self._ocr, "views", "base") != "base":
            selection_groups = groups if self.fg_backend == "classifier" else [[] for _ in paths]
            ocr_rows = self.select_candidates(ocr_rows, selection_groups)
            if self.fg_backend == "dictionary":
                groups = [get_func_groups(row.get("smiles")) if row.get("smiles") else []
                          for row in ocr_rows]

        validations = [validate_smiles(row.get("smiles")) for row in ocr_rows]
        canonical = [v["canonical"] if v["valid"] else None for v in validations]

        # IUPAC -------------------------------------------------------------
        iupac_started = time.perf_counter()
        names_sources = (self.resolver().resolve_many(canonical, workers=self.iupac_workers)
                         if use_iupac else [(None, None)] * len(paths))
        iupac_time = time.perf_counter() - iupac_started

        # If the selected view has no PubChem record, try the other views.
        if use_iupac:
            missing_idx = [i for i, (name, _) in enumerate(names_sources) if not name]
            if missing_idx:
                alt_pairs = []
                for i in missing_idx:
                    seen = {canonical[i]}
                    candidates = sorted(
                        ocr_rows[i].get("candidates") or [],
                        key=lambda c: (c.get("view") != "base",
                                       -(c.get("confidence") or 0.0)))
                    for candidate in candidates:
                        smiles = candidate.get("smiles")
                        value = (validate_smiles(smiles)["canonical"]
                                 if smiles else None)
                        if value and value not in seen:
                            seen.add(value)
                            alt_pairs.append((i, value))
                if alt_pairs:
                    unique = list(dict.fromkeys(value for _, value in alt_pairs))
                    alt_results = dict(zip(
                        unique, self.resolver().resolve_many(
                            unique, workers=self.iupac_workers)))
                    recovered = set()
                    for i, value in alt_pairs:
                        if names_sources[i][0]:
                            continue
                        name, source = alt_results.get(value, (None, None))
                        if name:
                            canonical[i] = value
                            validations[i] = validate_smiles(value)
                            ocr_rows[i]["smiles"] = value
                            names_sources[i] = (name, source)
                            recovered.add(i)
                    if self.fg_backend == "dictionary" and recovered:
                        for i in recovered:
                            groups[i] = get_func_groups(canonical[i])

        # Description -------------------------------------------------------
        desc_started = time.perf_counter()
        descriptions = (self.describer().describe_many(canonical)
                        if use_description else [None] * len(paths))
        desc_time = time.perf_counter() - desc_started

        results = []
        for i, image in enumerate(paths):
            row = ocr_rows[i]
            validation = validations[i]
            name, source = names_sources[i]
            desc = descriptions[i]

            iupac = dict(name=name, source=source,
                         status="ok" if name else ("miss" if use_iupac else "skipped"))
            if name and check_iupac and validation["valid"]:
                iupac["verification"] = verify_name(
                    name, validation["canonical"], jar=self._opsin_jar)

            if not use_description:
                desc_status = "skipped"
            elif desc:
                desc_status = "ok"
            elif not validation["valid"]:
                desc_status = "invalid_smiles"
            else:
                desc_status = "failed"

            results.append(dict(
                image=str(image),
                smiles=row.get("smiles"),
                raw_smiles=row.get("raw"),
                canonical_smiles=validation["canonical"],
                valid=validation["valid"],
                functional_groups=groups[i],
                iupac=iupac,
                description=dict(text=desc, status=desc_status),
                ocsr=dict(error=row.get("error"), latency_s=row.get("latency_s"),
                          confidence=row.get("selected_confidence",
                                             row.get("confidence")),
                          view=row.get("selected_view", row.get("view"))),
                timings_s=dict(
                    ocsr=round(ocr_time / len(paths), 3),
                    functional_groups=round(fg_time / len(paths), 3),
                    iupac=round(iupac_time / len(paths), 3),
                    description=round(desc_time / len(paths), 3),
                ),
            ))
        return results
