"""
从 Demo 模块导出 PPT 用素材图片（环境布局、范式对比、想象规划）
放到 DV/slides/ 目录
"""

import sys, os
from pathlib import Path

DV_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DV_DIR))
sys.path.insert(0, str(DV_DIR.parent))

from demo.app import render_home_layout, render_paradigm_comparison, generate_imagination_demo

out = Path(__file__).parent.parent / "slides"
out.mkdir(parents=True, exist_ok=True)

# 1. 环境布局
img = render_home_layout("small")
from PIL import Image
Image.fromarray(img).save(out / "env_layout.png")
print(f"[OK] env_layout.png")

# 2. 范式对比
img = render_paradigm_comparison()
if isinstance(img, Image.Image):
    img.save(out / "paradigm_comparison.png")
else:
    Image.fromarray(img).save(out / "paradigm_comparison.png")
print(f"[OK] paradigm_comparison.png")

# 3. 想象规划
img = generate_imagination_demo(n_branches=4, horizon=5)
Image.fromarray(img).save(out / "imagination.png")
print(f"[OK] imagination.png")

print(f"\nAll slides assets saved to: {out}")
