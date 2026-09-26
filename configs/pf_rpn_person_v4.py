"""人形微调 v4：把"中心度头卡死在初始值"的坑填掉。

v3 做对了一件事（``cls_pos_only``），但也把三个版本的**共同死穴**原样留着：
``sigmoid(ctr)`` 永远是常数 0.5。这一版专门修它。

为什么中心度头一步都没动（v3 实测）
------------------------------------
``tools/_scratch/probe_ctr.py`` 把三个假设依次证伪：

* **目标没信息**——证伪。匹配对的 centerness 目标中位 0.431、均值 0.448、
  范围 0.001~0.993，方差充足。
* **头收不到梯度**——证伪。``cgqs.net.*`` 收到全模型 5.85% 的梯度。
* **那是因为损失算错了**：``ctr_pos_only`` 分支里写的是
  ``(ctr_pred - ctr_t_match).abs().sum() / N``，**分母是全部 N=100 个 query**。
  负样本目标虽为 0，``ctr_pred`` 却不精确等于 0（有 ±0.05 残余），于是它们
  贡献了 loss_ctr 约 80% 的数值；而 L1 在 ``ctr_pred`` 附近的梯度符号取决于
  那一小撮残余的正负，**纯随机**。实测 32 个 batch 里 ``cgqs.net.2.bias``
  的梯度符号同号比例只有 **0.44（≈抛硬币）**，Adam 按符号更新 ⇒ 净漂移 ≈ 0
  ⇒ 训练完 ``cgqs.net.2.bias`` 仍是初始值 **-0.0504**，一步没动。

修法：只统计匹配成功的正样本，并按**正样本数**归一化（FCOS / ATSS 的标准做法）。
改完同号比例立刻变成 **1.00**，梯度恒为 -5.0，中心度头这才推得动。

``lam_ctr`` 由 5.0 降到 1.0：归一化方式一改，loss_ctr 的量级会涨约 18 倍
（实测 0.2055 → 3.6847），不回调会盖过 box 回归项。
"""

cfg = dict(
    # ---------------- 模型骨架（与 v1/v2/v3 完全一致）----------------
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

    # ---------------- 损失：相对 v3 的两处变化 ----------------
    loss=dict(
        loss_bbox=5.0,
        loss_giou=2.0,
        lam_ctr=1.0,            # v3 是 5.0（归一化方式改了，必须回调）
        loss_moe=1.0,
        cls_pos_weight=1.0,
        cls_neg_weight=1.0,
        use_moe=True,
        use_focal=True,
        focal_alpha=0.25,
        focal_gamma=2.0,
        ctr_pos_only=True,      # 同时决定"只算正样本"与"按正样本数归一化"
        cls_pos_only=True,      # v3 的关键改动，保留
    ),

    # ---------------- 数据（与 v1/v2/v3 完全一致）----------------
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
    out_dir="outputs_person_v4",
    seed=0,
)
