"""CPU 快速版配置：优先「能跑通」，参数按论文默认值保留，规模缩小。

与原版的对应关系见 README「配置对照表」。想要贴近论文规模，用 configs/pf_rpn_full.py。
"""

cfg = dict(
    # ---------------- 模型骨架 ----------------
    d_model=192,          # 论文 Grounding DINO 用 256
    nhead=6,
    num_layers=3,         # 论文 6 层
    dim_ff=384,
    num_queries=100,      # 论文 900（评测取 100/300/900 三个子集）
    dropout=0.1,
    backbone=dict(in_ch=3, widths=[32, 64, 128, 256], strides=[4, 2, 2, 2]),

    # ---------------- 三大核心模块 ----------------
    sia_topk=2,           # 论文 Table 7 最优 k=2
    csp_delta=0.3,        # 论文 Eq.2 阈值 delta=0.3
    csp_iters=3,          # 论文 Table 9 最优 3
    ctr_hidden=256,

    # ---------------- 损失（论文 Eq.5）----------------
    loss=dict(
        loss_bbox=5.0,    # L1 权重（同 DETR/GDINO）
        loss_giou=2.0,    # GIoU 权重
        lam_ctr=5.0,      # 论文 Table 8 最优 lambda=5
        loss_moe=1.0,     # L_rt 权重
        cls_pos_weight=1.0,
        cls_neg_weight=0.2,
        use_moe=True,
    ),

    # ---------------- 数据 ----------------
    data=dict(
        root="data",
        img_size=224,
        max_objs=4,
        n_images=512,
        train_n=512,
        val_n=128,
        # ⚠️ 合成数据的 val 图 id 是独立编号的 0..127，不要写 val_offset。
        # train.py 的兜底 ``id_offset = val_offset or (train_n if json_file else 0)``
        # 已经处理了，这里显式写 0 只是把语义钉死，防止有人"顺手"改成 train_n
        # 把验证集切空。
        val_offset=0,
        seed=0,
    ),

    # ---------------- 训练 ----------------
    optimizer=dict(lr=1e-4, weight_decay=1e-4),
    epochs=4,
    batch_size=2,
    num_workers=0,
    eval_every=1,
    topk_list=(100, 300),
    device="cpu",
    out_dir="outputs",
    seed=0,
)
