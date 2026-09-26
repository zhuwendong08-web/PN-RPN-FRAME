"""人形微调 v2：只改损失，其它一律不动画。

为什么要有 v2
-------------
v1（``pf_rpn_person.py``）训出来的模型，用 ``tools/diag_score.py`` 一测就露馅：

===============================  ==========================
指标                              v1 实测
===============================  ==========================
每图 logit>0 的 query 数          **恒为 0**（共 100 个）
分类分 sigmoid 范围              (0.009, 0.484) 全在 0.5 以下
中心度 sigmoid 范围              (0.487, 0.517) **几乎是个常数**
top1 落在几个不同 query 序号上    **只有 4 个**，最高频占 74%
top1 IoU 中位数                   0.000（三种打分口径全是 0）
===============================  ==========================

读法：分数已经失去区分度——``sigmoid(clf)*sigmoid(ctr)`` 里中心度是个常数
0.5，分类分全部为负，于是 argmax 实际在比"哪个 query 的下界最大"。
而 query 序号是固定的先验，所以**每张图挑中的都是同一批 query**，
落点自然一致地挤在某个角落。

这不是定位能力问题：同一批提案里，最佳 IoU 中位数有 0.660、74.5% 的图
至少有一个框命中人——**模型找得到人，只是排不了序**。

v2 只动两处 ``PFRPNLoss``：
1. ``use_focal=True``：极不平衡（约 4 正 : 96 负）下普通加权 BCE 会把
   全部 logit 压成负数，focal 的 ``(1-p_t)^gamma`` 压掉易分负样本，正样本
   才抬得起来。
2. ``ctr_pos_only=True``：负样本的中心度目标恒为 0，把它们算进均值会把
   中心度头整体推到原始输出 0，``sigmoid(0)=0.5`` 于是变成常乘子。
   FCOS / ATSS 都只在正样本上监督 centerness。
"""
cfg = dict(
    # ---------------- 模型骨架（与 pf_rpn_tiny.py、v1 完全一致）----------------
    d_model=192,
    nhead=6,
    num_layers=3,
    dim_ff=384,
    num_queries=100,
    dropout=0.1,
    backbone=dict(in_ch=3, widths=[32, 64, 128, 256], strides=[4, 2, 2, 2]),

    # ---------------- 三大核心模块 ----------------
    sia_topk=2,
    csp_delta=0.3,
    csp_iters=3,
    ctr_hidden=256,

    # ---------------- 损失：相对 v1 的唯一变化 ----------------
    loss=dict(
        loss_bbox=5.0,
        loss_giou=2.0,
        lam_ctr=5.0,
        loss_moe=1.0,
        cls_pos_weight=1.0,
        cls_neg_weight=0.2,
        use_moe=True,
        use_focal=True,        # v1 没有
        focal_alpha=0.25,
        focal_gamma=2.0,
        ctr_pos_only=True,     # v1 没有
    ),

    # ---------------- 数据（与 v1 完全一致）----------------
    data=dict(
        root="data",
        json_file="data/coco_person_mini.json",
        img_dir="data/coco/imgs",
        bbox_fmt="xywh_px",
        img_size=224,
        max_objs=4,
        train_n=1500,
        train_offset=0,
        val_n=200,
        val_offset=1500,
        seed=0,
    ),

    # ---------------- 训练 ----------------
    optimizer=dict(lr=1e-4, weight_decay=1e-4),
    epochs=2,
    batch_size=2,
    num_workers=0,
    eval_every=1,
    topk_list=(100, 300),
    device="cpu",
    out_dir="outputs_person_v2",
    seed=0,
)
