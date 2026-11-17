from .decoder import TransformerDecoder
from .embedding import Embeddings

try:  # vendored Swin is only needed for the old `swin_base`/`swin_large` names
    from .swin_transformer import swin_base, swin_large
except Exception:  # noqa: BLE001
    swin_base = swin_large = None
