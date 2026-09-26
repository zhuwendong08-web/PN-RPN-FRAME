"""PF-RPN（Prompt-Free Universal Region Proposal Network）精简复现包。

只想跑起来不读文档的话：

    from pfrpn.api import PF
    pf = PF()                                    # 自动找权重
    boxes, scores = pf.predict_file("a.jpg", topk=10)

命令行：``python -m pfrpn --help``。
"""
from .model import PFRPN
from .losses import PFRPNLoss
from .backbone import TinyMultiScaleBackbone
from .decoder import TransformerDecoder
from .modules.cgqs import CenternessScorer
from .modules.csp import CascadeSelfPrompt
from .modules.sia import SparseImageAwareAdapter

try:                                # api 依赖 numpy/PIL，缺了只是命令行不能用
    from .api import PF, Pred, list_checkpoints, to_pixels, from_pixels
    _HAS_API = True
except Exception as _e:             # pragma: no cover
    PF = None
    _HAS_API = False
    _API_ERR = _e

__all__ = [
    "PFRPN",
    "PFRPNLoss",
    "TinyMultiScaleBackbone",
    "TransformerDecoder",
    "SparseImageAwareAdapter",
    "CascadeSelfPrompt",
    "CenternessScorer",
    "PF",
    "Pred",
    "list_checkpoints",
    "to_pixels",
    "from_pixels",
]
