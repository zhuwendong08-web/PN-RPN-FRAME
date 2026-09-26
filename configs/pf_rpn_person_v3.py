"""人形微调 v3：分类损失改成"只在正样本上算"。

在 v2 的基础上只改一处 ``PFRPNLoss`` 开关（``cls_pos_only=True``），
其余（骨架 / 数据 / 训练超参 / 学习率）一律不动，便于直接对比。

为什么 v2 还不够
----------------
v2 动了 focal 与 ``ctr_pos_only``，AP50 从 0.0088 抬到 0.0122、top1 IoU
中位数从 0.000 抬到 0.089，但 ``tools/diag_score.py`` 报的硬判据**没动**：

===============================  ==========================
指标                              v1      v2
===============================  ==========================
每图 logit>0 的 query 数          0        **0**（共 100 个）
中心度 sigmoid 范围              ~0.5 常数  ~0.5 常数
top1 落在几个 query 序号上        4        4（最高频 74%）
===============================  ==========================

也就是说 focal 只是把负样本压得更低，**没有把正样本抬起来**。原因很清楚：
一张图约 4 正 : 96 负，focal 的 ``alpha=0.25`` 只是"少压一点负样本"，
负样本仍然占了损失的 96%。要真正提正样本的占比，只有一条路——
**不让负样本参与损失**。DETR / ATSS 都是这么做的。

同时把 ``cls_neg_weight`` 由 0.2 提到 1.0。负样本不再进损失后，这个参数
对梯度已经没有影响，这里只是把它还原成默认口径，避免以后误读成"v3 又改了权重"。
"""

cfg = dict(
    # ---------------- 模型骨架（与 pf_rpn_tiny.py、v1/v2 完全一致）----------------
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

    # ---------------- 损失：相对 v2 的唯一变化 ----------------
    loss=dict(
        loss_bbox=5.0,
        loss_giou=2.0,
        lam_ctr=5.0,
        loss_moe=1.0,
        cls_pos_weight=1.0,
        cls_neg_weight=1.0,        # v2 是 0.2
        use_moe=True,
        use_focal=True,            # v2 保留（v3 实际不走这条分支）
        focal_alpha=0.25,
        focal_gamma=2.0,
        ctr_pos_only=True,         # v2 保留
        cls_pos_only=True,         # v2 没有 —— v3 的关键改动
    ),

    # ---------------- 数据（与 v1/v2 完全一致）----------------
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
    out_dir="outputs_person_v3",
    seed=0,
)
