"""Generate README showcase figures for home-nav-dreamer.

Produces three self-contained diagrams (no JAX / Gradio required):
  - env_layout.png          : HomeNav-v0 grid layout with legend
  - paradigm_comparison.png : traditional modular vs. world-model architecture
  - imagination.png         : multi-branch imagination planning schematic
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrow

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, '..', 'assets')
os.makedirs(OUT, exist_ok=True)

HOME_LAYOUT_SMALL = [
    "RRR..........",
    "RRR..........",
    "RRR......EE..",
    "..........EE..",
    "..........EE..",
    "..........EE..",
    "..........EE..",
    "#####........",
    "..........GGG.",
    "..........GGG.",
    "..........GGG.",
]


def fig_env_layout():
    grid = HOME_LAYOUT_SMALL
    h, w = len(grid), len(grid[0])
    color = {
        'R': '#7eb0e6',   # room 1 (start)
        'G': '#8fd6a0',   # room 2 (goal)
        'E': '#f2c27d',   # elevator
        '#': '#5b4a42',   # wall
        '.': '#f4f4f8',   # corridor
    }
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for i in range(h):
        for j in range(w):
            ch = grid[i][j]
            c = color.get(ch, '#f4f4f8')
            ax.add_patch(plt.Rectangle((j, h - 1 - i), 1, 1,
                                       facecolor=c, edgecolor='#d5d5de',
                                       linewidth=1.2))
    # labels
    ax.text(3.5, h - 1.5, 'Room 1\n(Start)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#22406e')
    ax.text(7.5, h - 4.5, 'Elevator', ha='center', va='center',
            fontsize=10, fontweight='bold', color='#8a5a1a')
    ax.text(11.5, h - 0.7, 'Room 2\n(Goal)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1e5c2c')
    # agent path arrow (Room1 -> corridor -> elevator -> corridor -> Room2)
    ax.annotate('', xy=(7.0, h - 4.0), xytext=(2.5, h - 1.0),
                arrowprops=dict(arrowstyle='-|>', color='#d64545', lw=2.2))
    ax.annotate('', xy=(11.0, h - 1.0), xytext=(7.5, h - 5.0),
                arrowprops=dict(arrowstyle='-|>', color='#d64545', lw=2.2))
    ax.text(5.0, h - 2.2, 'agent path', fontsize=9, color='#d64545',
            style='italic')
    ax.set_xlim(-0.5, w + 0.5)
    ax.set_ylim(-0.5, h + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('HomeNav-v0 Grid Environment', fontsize=13, fontweight='bold',
                 pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'env_layout.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[OK] env_layout.png')


def fig_paradigm():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6.8))
    mods1 = ['LiDAR SLAM', 'AMCL\nLocalization', 'A* Global\nPlanner',
             'DWA Local\nPlanner', 'YOLO\nDetection', 'Depth\nEstimation',
             'State Machine\nScheduler']
    c1 = ['#8db8e8'] * 4 + ['#9fd6a6'] * 2 + ['#f2c27d']
    ax1.barh(range(len(mods1)), [1] * len(mods1), color=c1, height=0.62)
    for i, m in enumerate(mods1):
        ax1.text(0.5, i, m, ha='center', va='center', fontsize=9.5,
                 fontweight='bold', color='#222')
    ax1.set_xlim(0, 1); ax1.axis('off')
    ax1.set_ylim(-1.3, len(mods1) - 0.4)
    ax1.set_title('Traditional Modular Pipeline', fontsize=12.5,
                  fontweight='bold', color='#555')
    ax1.text(0.5, -1.08, 'Separate modules, cumulative error',
             transform=ax1.transAxes, ha='center', fontsize=11, color='#c0392b',
             fontweight='bold')

    mods2 = ['CNN Encoder', 'RSSM\nWorld Model', 'Decoder\n(Prediction)',
             'Reward Head', 'Policy Head\n(Action)', 'Value Head',
             'Latent\nImagination']
    c2 = ['#8db8e8', '#c39bd3', '#c39bd3', '#9fd6a6', '#f28b66',
          '#f28b66', '#f2d45d']
    ax2.barh(range(len(mods2)), [1] * len(mods2), color=c2, height=0.62)
    for i, m in enumerate(mods2):
        ax2.text(0.5, i, m, ha='center', va='center', fontsize=9.5,
                 fontweight='bold', color='#222')
    ax2.set_xlim(0, 1); ax2.axis('off')
    ax2.set_ylim(-1.3, len(mods2) - 0.4)
    ax2.set_title('World-Model End-to-End (DreamerV3)', fontsize=12.5,
                  fontweight='bold', color='#333')
    ax2.text(0.5, -1.08, 'Unified latent representation, end-to-end',
             transform=ax2.transAxes, ha='center', fontsize=11, color='#27ae60',
             fontweight='bold')
    fig.subplots_adjust(wspace=0.05)
    fig.savefig(os.path.join(OUT, 'paradigm_comparison.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[OK] paradigm_comparison.png')


def fig_imagination():
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    np.random.seed(7)
    origin = (1.0, 1.0)
    branches = [
        ('A: random actions', 'gray', 0.15, -0.5),
        ('B: optimal plan (selected)', 'green', 5.2, 8.5),
        ('C: cautious path', 'orange', 6.6, 3.2),
        ('D: failed (collision)', 'red', 8.2, -1.8),
    ]
    for name, color, angle, r in branches:
        t = np.linspace(0, 1, 40)
        rad = np.radians(angle)
        x = origin[0] + t * 4.2 * np.cos(rad) + np.random.normal(0, 0.08, 40)
        y = origin[1] + t * 4.2 * np.sin(rad) + np.random.normal(0, 0.08, 40)
        ax.plot(x, y, color=color, lw=2.4, alpha=0.85)
        ax.scatter(x[-1], y[-1], s=90, color=color, zorder=5, edgecolor='white')
        ax.text(x[-1] + 0.15, y[-1] + 0.1, f'{name}\nR={r:+.1f}',
                fontsize=9.5, color=color, fontweight='bold', va='center')
    ax.scatter(*origin, s=180, color='#34495e', zorder=6, edgecolor='white')
    ax.text(origin[0] - 0.3, origin[1] - 0.45, 'current\nstate', fontsize=9.5,
            ha='center', color='#34495e', fontweight='bold')
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-2.6, 4.6)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Multi-Branch Imagination Planning in Latent Space',
                 fontsize=13, fontweight='bold', pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'imagination.png'), dpi=150,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('[OK] imagination.png')


if __name__ == '__main__':
    fig_env_layout()
    fig_paradigm()
    fig_imagination()
    print('Done. Output dir:', os.path.abspath(OUT))
