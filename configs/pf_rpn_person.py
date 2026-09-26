"""人形微调配置：用 COCO person 子集替掉合成数据。

与 pf_rpn_tiny.py 的关系
------------------------
模型结构、超参、损失权重一律不动，只把数据换掉——这样实验变量单一，
「指标变化」才能干净地归因到「数据从合成换成真人」这一件事上。

数据从哪来
----------
COCO 每 ``file_name`` 都有独立 URL，所以**不必下整个 18.2 GB 的 train2017**：
``python tools/fetch_coco.py --n 1700 --out data/coco/imgs``
只下实际要用的那些图（1700 张约 300 MB，十来分钟）。
**必须比 ``train_n + val_n`` 多下 200 张**，否则验证集那部分图不存在，
跑起来会在读图时报错。
标注已由 ``tools/fetch_coco.py`` 之外的一次性脚本生成在 ``data/coco_person_mini.json``
（64,115 图 / 262,465 个 person 框的**全量**子集）。

检出效果不好是正常的：1500 张、224² 输入、纯 CPU，这三点叠加本来就不够。
关键是看 AR/AP 相对合成数据有没有**量级上的变化**。
"""

cfg = dict(
    # ---------------- 模型骨架（与 pf_rpn_tiny.py 完全一致）----------------
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

    # ---------------- 损失（不动，保持与合成数据同一套目标）----------------
    loss=dict(
        loss_bbox=5.0,
        loss_giou=2.0,
        lam_ctr=5.0,
        loss_moe=1.0,
        cls_pos_weight=1.0,
        cls_neg_weight=0.2,
        use_moe=True,
    ),

    # ---------------- 数据（这次唯一的变量）----------------
    data=dict(
        root="data",
        json_file="data/coco_person_mini.json",   # 不设就会去读合成 train.json
        img_dir="data/coco/imgs",                 # 按需下载的图在这里
        bbox_fmt="xywh_px",                       # COCO 官方口径
        img_size=224,
        max_objs=4,          # COCO person 子集每图平均 4.09 人，正好吻合
        train_n=1500,        # 只取前 1500 张；全量 64,115 张单 epoch 要 6.7 小时
        train_offset=0,
        val_n=200,           # 取 [1500:1700]，与训练集零重叠
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
    out_dir="outputs_person",      # 单独输出，绝不覆盖 outputs/ 里的合成成果
    seed=0,
)
