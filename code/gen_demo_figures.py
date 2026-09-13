"""Generate English demo figures for the paper.

These figures do NOT require training data - they are generated from code alone:
- paradigm_comparison.png: architecture comparison diagram
- env_layout.png: HomeNav-v0 layout visualization
- imagination.png: multi-branch imagination visualization
"""
import sys, os
from pathlib import Path

BASE = Path(__file__).parent.parent  # final_submission/root
sys.path.insert(0, str(BASE / "code"))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

OUT = BASE / "paper" / "figures"
os.makedirs(OUT, exist_ok=True)

# These require the app demo module which may need JAX
# If JAX is not available, generate placeholder images
try:
    from app import render_home_layout, render_paradigm_comparison, generate_imagination_demo

    print("Generating paradigm_comparison.png ...")
    img = render_paradigm_comparison()
    img.save(str(OUT / "paradigm_comparison.png"), "PNG")

    print("Generating env_layout.png ...")
    arr = render_home_layout(layout="small")
    img = Image.fromarray(arr)
    img.save(str(OUT / "env_layout.png"), "PNG")

    print("Generating imagination.png ...")
    arr = generate_imagination_demo(base_frame=None, n_branches=4, horizon=5)
    img = Image.fromarray(arr)
    img.save(str(OUT / "imagination.png"), "PNG")
    print("All demo figures generated successfully!")

except ImportError as e:
    print(f"Cannot generate figures (missing dependency: {e})")
    print("Run 'python gen_figures.py' after installing JAX to generate training curve figures.")
    print("Run this script after installing all dependencies to generate demo figures.")
