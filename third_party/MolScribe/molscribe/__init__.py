"""Lazy package entry point.

The upstream package imports ``.interface`` here, which pulls in
``albumentations`` through ``dataset.py``/``augment.py``. Inference only needs
``model``/``tokenizer``/``chemistry``, and those are independent of the
augmentation stack, so the import is deferred.
"""


def __getattr__(name):
    if name == "MolScribe":
        from .interface import MolScribe

        return MolScribe
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
