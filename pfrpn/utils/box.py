"""几何工具：框格式转换、IoU/GIoU、中心度目标、匈牙利匹配。

约定
----
* ``cxcywh``: (center_x, center_y, width, height)，归一化到 [0,1]（相对图像尺寸）
* ``xyxy``  : (left, top, right, bottom)，同样归一化到 [0,1]
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F

EPS = 1e-6


# --------------------------------------------------------------------------- #
# 格式转换
# --------------------------------------------------------------------------- #
def box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (..., 4)"""
    x, y, w, h = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], dim=-1)


def box_xyxy_to_cxcywh(boxes: torch.Tensor) -> torch.Tensor:
    """(..., 4) -> (..., 4)"""
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    return torch.stack([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1], dim=-1)


def _area(boxes_xyxy: torch.Tensor) -> torch.Tensor:
    w = (boxes_xyxy[..., 2] - boxes_xyxy[..., 0]).clamp(min=0)
    h = (boxes_xyxy[..., 3] - boxes_xyxy[..., 1]).clamp(min=0)
    return w * h


def _inter(boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor) -> torch.Tensor:
    """(N,4) x (M,4) -> (N,M)"""
    lt = torch.max(boxes1_xyxy[:, None, :2], boxes2_xyxy[None, :, :2])
    rb = torch.min(boxes1_xyxy[:, None, 2:], boxes2_xyxy[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    return wh[..., 0] * wh[..., 1]


def box_iou(boxes1_cxcywh: torch.Tensor, boxes2_cxcywh: torch.Tensor) -> torch.Tensor:
    """(N,4) x (M,4) -> (N,M)，输入为 cxcywh"""
    b1 = box_cxcywh_to_xyxy(boxes1_cxcywh)
    b2 = box_cxcywh_to_xyxy(boxes2_cxcywh)
    inter = _inter(b1, b2)
    union = _area(b1).unsqueeze(1) + _area(b2).unsqueeze(0) - inter
    return inter / union.clamp(min=EPS)


def box_giou(boxes1_cxcywh: torch.Tensor, boxes2_cxcywh: torch.Tensor) -> torch.Tensor:
    """(N,4) x (M,4) -> (N,M)，输入为 cxcywh"""
    b1 = box_cxcywh_to_xyxy(boxes1_cxcywh)
    b2 = box_cxcywh_to_xyxy(boxes2_cxcywh)
    inter = _inter(b1, b2)
    union = _area(b1).unsqueeze(1) + _area(b2).unsqueeze(0) - inter
    iou = inter / union.clamp(min=EPS)

    # 最小闭包框
    lt = torch.min(b1[:, None, :2], b2[None, :, :2])
    rb = torch.max(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    area_c = wh[..., 0] * wh[..., 1]
    return iou - (area_c - union).clamp(min=0) / area_c.clamp(min=EPS)


# --------------------------------------------------------------------------- #
# 中心度（FCOS 风格，作用于「预测框中心」与「真实框」之间）
# --------------------------------------------------------------------------- #
def centerness_target(pred_cxcywh: torch.Tensor, gt_cxcywh: torch.Tensor) -> torch.Tensor:
    """论文式 (4): c = sqrt( min(l,r)/max(l,r) * min(t,b)/max(t,b) )

    l/r/t/b 取「预测框中心」到「真实框四条边」的距离。
    当预测框与真实框完全重合时 c = 1，越偏离中心 c 越接近 0。

    返回 (N, M)：每行对应一个预测 query，每列对应一个 GT（未匹配列填 0）。
    """
    # 预测框中心；gt 的四条边
    # 注意：一律用切片索引再加维度，避免 pred[:, None, 0] 这类隐式索引歧义
    cx = pred_cxcywh[:, 0:1]                                  # (N,1)
    cy = pred_cxcywh[:, 1:2]                                  # (N,1)
    # 真实框的四条边（cxcywh -> xyxy）。注意要补成 (1,M) 才能与 (N,1) 广播成 (N,M)
    gx1 = gt_cxcywh[:, 0].reshape(1, -1) - gt_cxcywh[:, 2].reshape(1, -1) / 2   # (1,M)
    gy1 = gt_cxcywh[:, 1].reshape(1, -1) - gt_cxcywh[:, 3].reshape(1, -1) / 2
    gx2 = gt_cxcywh[:, 0].reshape(1, -1) + gt_cxcywh[:, 2].reshape(1, -1) / 2
    gy2 = gt_cxcywh[:, 1].reshape(1, -1) + gt_cxcywh[:, 3].reshape(1, -1) / 2

    l = (cx - gx1).abs()
    r = (gx2 - cx).abs()
    t = (cy - gy1).abs()
    b = (gy2 - cy).abs()

    def _ratio(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        denom = torch.maximum(a, c)
        num = torch.minimum(a, c)
        return torch.where(denom < EPS, torch.ones_like(denom), num / denom.clamp(min=EPS))

    prod = _ratio(l, r) * _ratio(t, b)

    # ⚠️ 开方这一步必须防反向。漏了它埋的雷是：
    #   * ``_ratio`` 的最小值是 0（预测框中心正好压在真实框某条边上时），
    #     只要两个 ratio 中有一个是 0，乘积 ``prod`` 就是 0；
    #   * ``sqrt`` 的反向是 ``grad / (2*sqrt(x))``，x=0 时退化成 0/0（或 ±inf）——
    #     正向 ``sqrt(0)=0`` 完全看不出问题，反向却产出 NaN。
    # 这一步一 NaN，`clip_grad_norm_` 会把它按"总范数异常"放大成**全部参数
    # 都是 NaN**，整个训练当场报废。v2 训练就是第 333 步撞上的，当时的报错是
    # ``ValueError: matrix contains invalid numeric entries``（scipy 拒绝非有限
    # 的代价矩阵），完全指不到根因。
    #
    # 注意不能直接写 ``sqrt(prod.clamp(min=EPS))``：那会把"贴边 => 中心度 0"
    # 这个正确语义改成 0.001（EPS=1e-6 开方后）。这里用 where 保住正向精确值，
    # 只在 prod<=0 的位置把梯度遮成 0。
    pos = prod > 0
    safe = torch.where(pos, prod, torch.full_like(prod, EPS))
    return torch.where(pos, torch.sqrt(safe), torch.zeros_like(prod))


# --------------------------------------------------------------------------- #
# 匈牙利匹配
# --------------------------------------------------------------------------- #
def hungarian_match(cost: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """对 (N, M) 代价矩阵做最优匹配。

    返回 (rows, cols)，其中 cols[rows[k]] 表示第 k 个 query 匹配的 GT 下标。
    """
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError:  # 退化：贪心匹配
        n, m = cost.shape
        rows = torch.arange(n)
        cols = cost.argmin(dim=1)
        used = set()
        for i, c in enumerate(cols.tolist()):
            while c in used:
                c = int((cost[i] < cost[i, c]).nonzero().flatten()[-1]) if False else -1
                break
        return rows, cols

    n, m = cost.shape
    # 代价矩阵出现 NaN/inf 时 scipy 只丢一句
    # ``ValueError: matrix contains invalid numeric entries``——而真正的元凶
    # 通常在更早的地方（例如中心度目标的 sqrt 反向产出 NaN），把这句原样贴出来
    # 会让人白白怀疑数据。这里先按" NaN > inf > 正常 "排序替换成有限值，
    # 既让训练能继续，也把真实的非有限元素打印出来。
    arr = cost.detach().cpu().numpy()
    if not np.isfinite(arr).all():
        nan_n = int(np.isnan(arr).sum())
        inf_n = int(np.isinf(arr).sum())
        print(f"[hungarian] ⚠️ 代价矩阵有 {nan_n} 个 NaN / {inf_n} 个 inf "
              f"（形状 {arr.shape}）。已用有限值替换后继续，"
              f"但请先去查前向是否已有非有限输出——这里往往是下游症状。",
              flush=True)
        arr = np.where(np.isnan(arr), np.inf, arr)
        arr = np.where(np.isposinf(arr), 1e30, arr)
        arr = np.where(np.isneginf(arr), -1e30, arr)

    if n <= m:
        r, c = linear_sum_assignment(arr)
        return torch.as_tensor(r), torch.as_tensor(c)
    else:  # query 多于 GT：转置问题后翻转回来
        r, c = linear_sum_assignment(arr.T)
        return torch.as_tensor(c), torch.as_tensor(r)
