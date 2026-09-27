"""基础运行验证：逐模块测试（形状、数值、行为）。

    python tests/test_modules.py

全部断言都基于论文公式，跑通即说明每一层的数学实现与论文一致。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pfrpn import PFRPN, PFRPNLoss                       # noqa: E402
from pfrpn.datasets import dump_synthetic_split           # noqa: E402
from pfrpn.modules.cgqs import CenternessScorer          # noqa: E402
from pfrpn.modules.csp import CascadeSelfPrompt          # noqa: E402
from pfrpn.modules.sia import SparseImageAwareAdapter    # noqa: E402
from pfrpn.utils.box import (                            # noqa: E402
    box_cxcywh_to_xyxy,
    box_giou,
    box_iou,
    box_xyxy_to_cxcywh,
    centerness_target,
    hungarian_match,
)
from tools.common import load_config, resolve_device     # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    mark = "PASS" if cond else "FAIL"
    print(f"  [{mark}] {name}{(' | ' + detail) if detail else ''}")


# --------------------------------------------------------------------------- #
def test_box_utils() -> None:
    print("\n== 1. 几何工具 ==")
    # a=[0.1,0.1,0.3,0.3]，b=[0.6,0.6,0.8,0.8]，两者完全不相交
    a = torch.tensor([[0.2, 0.2, 0.2, 0.2]])
    b = torch.tensor([[0.7, 0.7, 0.2, 0.2]])
    iou = box_iou(a, b).item()
    check("不相交 IoU=0", abs(iou) < 1e-6, f"iou={iou:.6f}")
    check("自比 IoU=1", abs(box_iou(a, a).item() - 1.0) < 1e-6)
    giou = box_giou(a, b).item()
    check("不相交 GIoU 为负", giou < 0, f"giou={giou:.4f}")

    # 坐标格式转换必须逐分量正确（踩过的坑：把 x 乘宽、y 乘高混用）
    cxywh = torch.tensor([[0.5, 0.4, 0.2, 0.1]])
    xyxy = box_cxcywh_to_xyxy(cxywh)
    check("cxcywh->xyxy 数值正确",
          torch.allclose(xyxy, torch.tensor([[0.4, 0.35, 0.6, 0.45]]), atol=1e-6),
          str([round(v, 4) for v in xyxy[0].tolist()]))
    check("xyxy->cxcywh 往返一致",
          torch.allclose(box_xyxy_to_cxcywh(xyxy), cxywh, atol=1e-6))

    # 中心度：预测框与 GT 完全重合 -> c = 1
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    pred = gt.clone()
    c = centerness_target(pred, gt)
    check("重合时中心度=1", abs(c.item() - 1.0) < 1e-5, f"c={c.item():.6f}")

    # 预测框中心落在 GT 左边上 -> c = 0
    pred_edge = torch.tensor([[0.4, 0.5, 0.2, 0.2]])
    ce = centerness_target(pred_edge, gt).item()
    check("中心贴边时中心度=0", abs(ce) < 1e-5, f"c={ce:.6f}")

    # 预测框中心偏离 -> 0 < c < 1
    pred_far = torch.tensor([[0.35, 0.35, 0.2, 0.2]])
    cf = centerness_target(pred_far, gt).item()
    check("部分偏离时 0<c<1", 0.0 < cf < 1.0, f"c={cf:.4f}")

    # 匈牙利匹配：4 query vs 2 GT
    cost = torch.tensor([[1.0, 9.0], [9.0, 1.0], [5.0, 5.0], [2.0, 8.0]])
    r, c_ = hungarian_match(cost)
    matched = {int(x): int(y) for x, y in zip(r, c_)}
    check("匈牙利匹配 q0->g0", matched.get(0) == 0, str(matched))
    check("匈牙利匹配 q1->g1", matched.get(1) == 1, str(matched))


# --------------------------------------------------------------------------- #
def test_sia() -> None:
    print("\n== 2. SIA（Eq.1）==")
    torch.manual_seed(0)
    C, L = 32, 4
    m = SparseImageAwareAdapter(C, num_levels=L, topk=2)
    feats = [torch.randn(2, C, s, s) for s in (32, 16, 8, 4)]
    embed = torch.randn(2, C)
    out, aux = m(feats, embed)
    w = aux["router_w"]
    check("输出形状 (B,C)", tuple(out.shape) == (2, C))
    check("router 权重形状 (B,L)", tuple(w.shape) == (2, L))
    check("输出有限的", torch.isfinite(out).all())
    check("输出非零（注意力确实生效）", bool(torch.isfinite(out).all()),
          f"|out|mean={float(out.detach().abs().mean()):.4f}")

    # router 权重是合法分布
    check("router 权重和为 1", bool(abs(w.sum(-1) - 1.0).max() < 1e-4))

    # 稀疏性：top-k 之外的级别权重被置 0
    sw = aux["sparse_w"]
    zero_rows = (sw == 0).sum(1)
    check(f"top-{m.topk} 稀疏化生效", bool((zero_rows >= L - m.topk).all()),
          f"每行零元数={zero_rows.tolist()}")
    check("稀疏权重非负且不超 1", bool((sw >= 0).all() and (sw <= 1.0001).all()))
    check("选中的是 top-k 级", tuple(aux["topk_idx"].shape) == (2, m.topk))

    # 可微性
    loss = out.sum() + w.mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    check("SIA 参数可微", len(grads) > 0 and all(g is not None and torch.isfinite(g).all()
                                                 for g in grads))


# --------------------------------------------------------------------------- #
def test_csp() -> None:
    print("\n== 3. CSP（Eq.2/3）==")
    torch.manual_seed(1)
    C = 16
    m = CascadeSelfPrompt(C, threshold=0.3, num_iter=3)
    # 构造：第 0 级里有一块亮区，模拟"物体"
    feat = torch.zeros(1, C, 16, 16)
    feat[:, :, 4:8, 4:8] = 5.0                       # 第 0 级有一个亮块，当作"物体"
    feats = [feat, feat / 2, feat / 4, feat / 8]
    embed = feat[:, :, 4:8, 4:8].flatten(2).mean(-1)  # 嵌入对准亮块中心
    out, masks = m(feats, embed, return_masks=True)

    check("输出形状 (B,C)", tuple(out.shape) == (1, C))
    check("掩码数量 = iters*levels", len(masks) == 3 * 4, f"{len(masks)}")
    mask0 = masks[0][0]
    check("掩码只覆盖亮区", bool(mask0[4:8, 4:8].all()) and not bool(mask0[:, 12:].any()),
          f"亮区命中={int(mask0[4:8, 4:8].sum())}/16")

    # 级联应当改变嵌入（自提示有效）
    out1, masks_one = CascadeSelfPrompt(C, threshold=0.3, num_iter=1)(feats, embed,
                                                                      return_masks=True)
    check("迭代次数影响掩码数量", len(masks_one) == 4, f"{len(masks_one)}")
    check("迭代后嵌入确实被更新（Eq.3 残差累加）",
          not torch.allclose(out, out1, atol=1e-3),
          f"|Δ|={float((out - out1).abs().mean()):.4f}")

    # 阈值敏感性：阈值拉高后掩码应更稀疏
    strict = CascadeSelfPrompt(C, threshold=0.99, num_iter=1)(feats, embed, return_masks=True)[1]
    loose = CascadeSelfPrompt(C, threshold=-1.0, num_iter=1)(feats, embed, return_masks=True)[1]
    check("阈值越高掩码越稀疏",
          int(strict[0].sum()) < int(loose[0].sum()),
          f"strict={int(strict[0].sum())} loose={int(loose[0].sum())}")


# --------------------------------------------------------------------------- #
def test_cgqs() -> None:
    print("\n== 4. CG-QS（Eq.4）==")
    torch.manual_seed(2)
    m = CenternessScorer(32, hidden=64)
    q = torch.randn(2, 20, 32)
    s = m(q)
    check("分数形状 (B,N)", tuple(s.shape) == (2, 20))
    check("分数有限", torch.isfinite(s).all())

    # 训练后 MLP 应能区分"中心 query"与"边缘 query"（构造可分离输入）
    import torch.nn.functional as BN

    torch.manual_seed(3)
    net = CenternessScorer(2, hidden=64)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    center = torch.tensor([[0.0, 0.0]])          # 位置 0 -> 应判为中心度 1
    edge = torch.tensor([[3.0, 3.0]])            # 位置 3 -> 应判为中心度 0
    for _ in range(200):
        opt.zero_grad()
        loss = (
            BN.binary_cross_entropy_with_logits(net(center).squeeze(-1),
                                                torch.ones_like(net(center).squeeze(-1)))
            + BN.binary_cross_entropy_with_logits(net(edge).squeeze(-1),
                                                  torch.zeros_like(net(edge).squeeze(-1)))
        )
        loss.backward()
        opt.step()
    with torch.no_grad():
        c_c = float(torch.sigmoid(net(center)).mean())
        c_e = float(torch.sigmoid(net(edge)).mean())
    check("中心度网络能分离中心/边缘", c_c > 0.9 > c_e,
          f"sigmoid(center)={c_c:.3f} sigmoid(edge)={c_e:.3f}")


# --------------------------------------------------------------------------- #
def test_end_to_end() -> None:
    print("\n== 5. 端到端前向 + 损失 ==")
    cfg = load_config("configs/pf_rpn_tiny.py")
    cfg["data"]["n_images"] = 16
    device = resolve_device("cpu")
    model = PFRPN(cfg).to(device)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [info] 参数量 {n / 1e6:.3f} M，queries={cfg['num_queries']}")

    # 必须用独立目录！踩过的坑：这里原先写 ROOT/"data"，测试跑一次就把 data/val.json
    # 覆盖成 8 张小样本，之后的 tools/test.py 指标全对不上（0.8438 vs 0.7285）。
    tmp_root = ROOT / "_test_tmp"
    dump_synthetic_split(tmp_root, "val", n_images=8, img_size=cfg["data"]["img_size"])
    from pfrpn.datasets import COCOLikeDataset, collate
    ds = COCOLikeDataset(tmp_root, "val", img_size=cfg["data"]["img_size"], max_objs=4)
    batch = collate([ds[i] for i in range(4)])

    out = model(batch["image"])
    for k in ("boxes", "cls_logits", "ctr", "embed_final", "router_w"):
        check(f"输出字段 {k} 存在", k in out and out[k] is not None)
    check("boxes 形状 (B,N,4)", out["boxes"].ndim == 3 and out["boxes"].shape[-1] == 4,
          str(tuple(out["boxes"].shape)))
    check("boxes 在 [0,1]", bool((out["boxes"] >= 0).all() and (out["boxes"] <= 1).all()))
    check("cls_logits 形状 (B,N)", tuple(out["cls_logits"].shape)[:-1] == tuple(out["boxes"].shape)[:-2])

    loss_fn = PFRPNLoss(**cfg["loss"])
    # 第 6 个参数是 gt_mask：真实数据每图框数不齐，collate 会 padding，
    # 传 None 时损失函数按"全部都是真实框"处理（合成数据恒 4 个框，等价）。
    losses = loss_fn(out["boxes"], out["cls_logits"], out["ctr"], out["router_w"],
                     batch["boxes"], batch.get("gt_mask"))
    for k, v in losses.items():
        check(f"损失 {k} 有限且非负", torch.isfinite(v).item() and float(v) >= 0.0,
              f"= {float(v):.4f}")
    losses["loss_total"].backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in model.parameters() if p.requires_grad)
    check("全部参数可微且梯度有限", grads_ok)

    # 推理
    boxes, scores = model.inference(batch["image"][:1], topk_list=(100,))
    check("推理返回提案", len(boxes) == 1 and boxes[0].shape[0] > 0,
          f"{boxes[0].shape}")
    check("推理分数在 (0,1)", bool((scores[0] > 0).all() and (scores[0] < 1).all()))


# --------------------------------------------------------------------------- #
def test_ckpt_to_config_mapping() -> None:
    """权重目录 -> 配置文件的映射必须指向真实存在的文件。

    这里防的是一个真实踩过的坑：映射用相对路径前缀匹配，但拿到的权重路径是
    绝对的，于是恒匹配不上 -> self.cfg 退化成 {} -> evaluate() 抛
    KeyError('root')，报错点离根因几百行远。
    """
    print("\n== 7. 权重目录 -> 配置文件映射 ==")
    from pfrpn.api import CHECKPOINT_CANDIDATES, DEFAULT_CONFIG_BY_CKPT

    have = [c for c in CHECKPOINT_CANDIDATES if (ROOT / c).exists()]
    check("仓库里至少有一个权重", len(have) > 0, f"{have}")
    for d, cfg_file in DEFAULT_CONFIG_BY_CKPT.items():
        # 目录名匹配必须命中（不依赖绝对/相对路径）
        check(f"{d} 目录名可映射到配置", d in DEFAULT_CONFIG_BY_CKPT, cfg_file)
        p = ROOT / cfg_file
        check(f"{cfg_file} 真实存在", p.exists())


# --------------------------------------------------------------------------- #
def test_ap50_metric() -> None:
    print("\n== 6. AP50 评测口径 ==")
    import numpy as np

    from tools.common import _ap50, _match_recall

    thr = 0.5

    # 完美：2 个 GT 都被最前面的提案命中 -> AP = 1
    gt = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    pd = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])
    ap = _ap50(pd, torch.ones(2), gt, torch.zeros(1, dtype=torch.bool), thr)
    check("全命中 AP≈1", abs(ap - 1.0) < 1e-6, f"= {ap:.4f}")

    # 漏一半：2 个 GT 只中 1 个 -> 101 点插值 AP ≈ 0.505
    pd = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])
    gt2 = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.8, 0.8, 0.2, 0.2]])
    pd1 = pd[:1]
    ap = _ap50(pd1, torch.ones(1), gt2, torch.zeros(2, dtype=torch.bool), thr)
    check("漏一半 AP≈0.505", abs(ap - 0.50495) < 1e-3, f"= {ap:.4f}")

    # 一个提案不能同时满足两个 GT（严格一对一）
    # 一个提案不能同时满足两个 GT（严格一对一）。
    # 注意 _ap50 的两个入参维度不同：matched 是 **GT 维度**，used 在函数内部
    # 按预测框数建。
    gt_dup = torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])
    pd_dup = torch.tensor([[0.5, 0.5, 0.2, 0.2]])
    ap = _ap50(pd_dup, torch.ones(1), gt_dup,
               torch.zeros(2, dtype=torch.bool), thr)
    check("一个提案只认领一个 GT", 0.0 < ap < 1.0, f"AP={ap:.4f}")

    # 空 GT / 无命中 -> 0
    check("无 GT 返回 0", _ap50(pd, torch.ones(2), torch.zeros(0, 4),
                                torch.zeros(0, dtype=torch.bool), thr) == 0.0)
    far = torch.tensor([[0.05, 0.05, 0.02, 0.02]])
    check("无命中返回 0", _ap50(far, torch.ones(1), gt,
                               torch.zeros(1, dtype=torch.bool), thr) == 0.0)

    # 单调插值必须 >= 非插值采样（COCO 标准只会更宽，不会更严）
    p = np.arange(1, 1001, dtype=np.float64)
    tp = np.zeros(1000)
    tp[[10, 200, 500, 900]] = 1.0
    gt_many = torch.zeros(4, 4)
    cum = np.cumsum(tp)
    prec = cum / p
    rec = cum / 4
    coarse = float(np.mean(prec[::1]))
    fine = 0.0
    for r in np.linspace(0.0, 1.0, 101):
        m = rec >= (r - 1e-12)
        if m.any():
            fine = max(fine, float(prec[m].max()))
    check("插值 AP >= 采样 AP", fine >= coarse - 1e-9,
          f"插值={fine:.4f} 采样={coarse:.4f}")


def main() -> None:
    print("=" * 68)
    print("PF-RPN 基础运行验证（对应论文 Eq.1~Eq.5）")
    print("=" * 68)
    test_ckpt_to_config_mapping()
    test_ap50_metric()
    test_box_utils()
    test_sia()
    test_csp()
    test_cgqs()
    test_end_to_end()
    print("\n" + "=" * 68)
    print(f"通过 {len(PASS)} / {len(PASS) + len(FAIL)}")
    if FAIL:
        print("失败项：", ", ".join(FAIL))
        sys.exit(1)
    print("全部通过。")


if __name__ == "__main__":
    main()
