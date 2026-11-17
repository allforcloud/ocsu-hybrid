"""ConvNeXt-T multi-label functional-group classifier.

The checkpoint is published on Hugging Face (107 MB) and contains the weights,
the 140-name vocabulary, the input size and the tuned global threshold.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from PIL import Image

from .. import paths

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class FGClassifier:
    def __init__(self, checkpoint: str | Path | None = None,
                 device: str | None = None):
        self.checkpoint = Path(checkpoint) if checkpoint else paths.fg_ckpt()
        self.device_name = device
        self._model = None
        self._vocab: List[str] = []
        self._img_size = 384
        self._threshold = 0.5
        self._torch = None
        self._transform = None
        self.device = None

    # ------------------------------------------------------------------ load
    def load(self) -> "FGClassifier":
        if self._model is not None:
            return self
        import timm
        import torch
        from torchvision import transforms as T

        paths.require(
            self.checkpoint,
            "Download the functional-group head with `python scripts/download_models.py --fg` "
            "or pass --fg-ckpt / set OCSU_FG_CKPT.",
        )
        ckpt = torch.load(self.checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model") or ckpt.get("state_dict")
        if state is None:
            raise ValueError(f"{self.checkpoint} has no 'model' or 'state_dict' entry")
        self._vocab = list(ckpt["vocab"])
        self._img_size = int(ckpt.get("img_size", 384))
        self._threshold = float(ckpt.get("threshold", 0.5))

        self.device = torch.device(
            self.device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
        model = timm.create_model("convnext_tiny", pretrained=False,
                                  num_classes=len(self._vocab))
        model.load_state_dict(state)
        self._model = model.to(self.device).eval()
        self._transform = T.Compose([
            T.Resize((self._img_size, self._img_size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
        self._torch = torch
        return self

    @property
    def vocab(self) -> List[str]:
        self.load()
        return list(self._vocab)

    @property
    def threshold(self) -> float:
        self.load()
        return self._threshold

    # --------------------------------------------------------------- predict
    def predict_proba(self, images: Sequence[str | Path],
                      batch_size: int = 32) -> np.ndarray:
        """Return an ``(n_images, n_classes)`` probability matrix."""
        self.load()
        torch = self._torch
        if not images:
            return np.zeros((0, len(self._vocab)), dtype=np.float32)
        chunks = []
        with torch.inference_mode():
            for start in range(0, len(images), batch_size):
                batch = images[start:start + batch_size]
                tensors = []
                for path in batch:
                    with Image.open(path) as img:
                        tensors.append(self._transform(img.convert("RGB")))
                x = torch.stack(tensors).to(self.device)
                chunks.append(torch.sigmoid(self._model(x)).float().cpu().numpy())
        return np.concatenate(chunks, axis=0)

    def predict(self, images: Sequence[str | Path], batch_size: int = 32,
                threshold: Optional[float] = None) -> List[List[str]]:
        """Return the group names above the tuned threshold for each image."""
        probs = self.predict_proba(images, batch_size=batch_size)
        thr = self._threshold if threshold is None else float(threshold)
        return [[self._vocab[j] for j in range(len(self._vocab)) if row[j] >= thr]
                for row in probs]

    def predict_one(self, image: str | Path, threshold: Optional[float] = None
                    ) -> List[str]:
        return self.predict([image], batch_size=1, threshold=threshold)[0]
