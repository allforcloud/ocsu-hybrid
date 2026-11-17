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
                                 batch_size=molscribe_batch_size)
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

        validations = [validate_smiles(row.get("smiles")) for row in ocr_rows]
        canonical = [v["canonical"] if v["valid"] else None for v in validations]

        # Functional groups -------------------------------------------------
        fg_started = time.perf_counter()
        if self.fg_backend == "classifier":
            groups = self.fg().predict(paths, batch_size=self.fg_batch_size)
        else:
            groups = [get_func_groups(smiles) if smiles else [] for smiles in canonical]
        fg_time = time.perf_counter() - fg_started

        # IUPAC -------------------------------------------------------------
        iupac_started = time.perf_counter()
        names_sources = (self.resolver().resolve_many(canonical, workers=self.iupac_workers)
                         if use_iupac else [(None, None)] * len(paths))
        iupac_time = time.perf_counter() - iupac_started

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
                ocsr=dict(error=row.get("error"), latency_s=row.get("latency_s")),
                timings_s=dict(
                    ocsr=round(ocr_time / len(paths), 3),
                    functional_groups=round(fg_time / len(paths), 3),
                    iupac=round(iupac_time / len(paths), 3),
                    description=round(desc_time / len(paths), 3),
                ),
            ))
        return results
