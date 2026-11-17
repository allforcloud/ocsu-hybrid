"""OCSR backends. The shipped backend is MolScribe."""
from .molscribe import MolScribeOCR, recognize, recognize_batch

__all__ = ["MolScribeOCR", "recognize", "recognize_batch"]
