"""PF-RPN 最短上手：改下面两个变量就能跑。

运行：
    python examples/quick_start.py
"""
from pfrpn.api import PF

# ① 想测哪张图
IMAGE = "outputs/demo.png"
# ② 用哪个权重。
#    "outputs/best.pth"        合成数据上训的通用权重
#    "outputs_person_v4/best.pth"  微调过、专门检测人的权重
CKPT = "outputs/best.pth"

pf = PF(checkpoint=CKPT)
print(f"用的权重：{pf.checkpoint}")

# 推理：boxes 是 (N, 4)，scores 是 (N,)，都按分数从高到低排好
boxes, scores = pf.predict_file(IMAGE, topk=10)

print(f"{IMAGE} 出 {len(boxes)} 个提案：")
for i, (b, s) in enumerate(zip(boxes, scores)):
    print(f"  #{i + 1:<3} score={s:.4f}  box=[{b[0]:.3f},{b[1]:.3f},"
          f"{b[2]:.3f},{b[3]:.3f}]")

# 画框存图。topk 别开太大，分数挤在一起时前面几个看不出差别
out = pf.draw_file(IMAGE, "outputs/quick_start.png", topk=10, thr=0.0)
print(f"带框图 -> {out}")
