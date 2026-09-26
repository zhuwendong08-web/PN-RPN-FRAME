"""贴近原版规模的配置（需要 GPU，纯 CPU 上很慢）。

原版配置（来自 GitHub README）：
    Python 3.10 / CUDA 11.8 / PyTorch 2.1.0 / MMDetection 3.3.0 / Grounding DINO + Swin-B
    训练数据：COCO 2017 的 5% + ImageNet-1k 的 5%
    硬件：4 x RTX 4090
"""

cfg = dict(
    # ---------------- 模型骨架（对齐 Grounding DINO）----------------
    d_model=256,
    nhead=8,
    num_layers=6,
    dim_ff=1024,
    num_queries=900,      # 论文 AR 评估用的就是 900
    dropout=0.1,
    backbone=dict(in_ch=3, widths=[48, 96, 192, 384], strides=[4, 2, 2, 2]),

    # ---------------- 三大核心模块 ----------------
    sia_topk=2,
    csp_delta=0.3,
    csp_iters=3,
    ctr_hidden=256,

    # ---------------- 损失 ----------------
    loss=dict(
        loss_bbox=5.0,
        loss_giou=2.0,
        lam_ctr=5.0,
        loss_moe=1.0,
        cls_pos_weight=1.0,
        cls_neg_weight=0.2,
        use_moe=True,
    ),

    # ---------------- 数据（需要真实 COCO / ImageNet）----------------
    # 真实数据请自备；json 的 bbox 必须是 cxcywh 归一化格式
    data=dict(
        root="data/coco",
        img_size=512,
        max_objs=10,
        n_images=4000,
        train_n=4000,
        val_n=500,
        # 见 configs/pf_rpn_tiny.py 的说明：合成数据的 val 图不按 [train_n:]
        # 编排，偏移会把验证集切空。
        val_offset=0,
        seed=0,
        force_synth=True,     # 没有真实数据就混一份合成的，保证脚本能跑
    ),

    optimizer=dict(lr=1e-4, weight_decay=1e-4),
    epochs=12,
    batch_size=8,
    num_workers=2,
    eval_every=1,
    topk_list=(100, 300, 900),
    device="auto",          # 有 CUDA 就用 cuda，否则 cpu
    out_dir="outputs",
    seed=0,
)
