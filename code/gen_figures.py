
"""
Minimal script to generate paper figures from metrics.jsonl
Generates: training_score.png, training_loss.png, training_perf.png, training_dashboard.png
"""
import json, os, sys
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
METRICS = os.path.join(BASE, "..", "data", "metrics.jsonl")
OUT = os.path.join(BASE, "..", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

print(f"Metrics: {METRICS}")
print(f"Output:  {OUT}")

steps, scores, lengths = [], [], []
train_steps, rand_actions = [], []
dyn_losses, rep_losses, rew_losses = [], [], []
image_losses = []
fps_policy, fps_train = [], []

with open(METRICS, 'r', encoding='utf-8') as f:
    for line in f:
        d = json.loads(line)
        s = d.get('step')
        if s is None: continue
        steps.append(s)
        scores.append(d.get('episode/score', np.nan))
        lengths.append(d.get('episode/length', np.nan))
        if 'train/rand/action' in d:
            train_steps.append(s)
            rand_actions.append(d['train/rand/action'])
            dyn_losses.append(d['train/loss/dyn'])
            rep_losses.append(d['train/loss/rep'])
            rew_losses.append(d['train/loss/rew'])
            if 'train/loss/image' in d:
                image_losses.append(d['train/loss/image'])
            fps_policy.append(d['fps/policy'])
            fps_train.append(d['fps/train'])

steps = np.array(steps, dtype=float)
scores = np.array(scores, dtype=float)
lengths = np.array(lengths, dtype=float)
train_steps = np.array(train_steps, dtype=float)
rand_actions = np.array(rand_actions, dtype=float)
dyn_losses = np.array(dyn_losses, dtype=float)
rep_losses = np.array(rep_losses, dtype=float)
rew_losses = np.array(rew_losses, dtype=float)
image_losses = np.array(image_losses, dtype=float)
fps_policy = np.array(fps_policy, dtype=float)
fps_train = np.array(fps_train, dtype=float)

print(f"Loaded: {len(steps)} episode lines, {len(train_steps)} training lines")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as _fm

def _setup_matplotlib_font():
    _fm._load_fontmanager(try_read_cache=False)
    available = {f.name for f in _fm.fontManager.ttflist}
    candidates = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Noto Sans CJK SC']
    for fn in candidates:
        if fn in available:
            plt.rcParams['font.sans-serif'] = [fn, 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

_setup_matplotlib_font()
plt.rcParams.update({'font.size': 10})

# Figure 1: Score + Episode Length
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(steps, scores, color='#2196F3', lw=0.8, alpha=0.7)
if len(scores) > 100:
    w = max(len(scores)//30, 10)
    ma = np.convolve(np.nan_to_num(scores), np.ones(w)/w, mode='valid')
    ax1.plot(steps[w-1:], ma, color='#FF5722', lw=2, label=f'MA({w})')
ax1.axhline(y=0, color='red', ls='--', lw=0.8, alpha=0.5)
ax1.set_xlabel('Step'); ax1.set_ylabel('Score')
ax1.set_title('Episode Score'); ax1.grid(alpha=0.25); ax1.legend(fontsize=9)

ax2.plot(steps, lengths, color='#4CAF50', lw=0.8, alpha=0.7)
ax2.set_xlabel('Step'); ax2.set_ylabel('Steps')
ax2.set_title('Episode Length'); ax2.grid(alpha=0.25)
fig.suptitle(f'DreamerV3 HomeNav Training ({int(steps[-1])} steps)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'training_score.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] training_score.png")

# Figure 2: World Model Losses
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
configs = [
    (0,0, dyn_losses, 'Dynamic KL Loss', '#9C27B0'),
    (0,1, rep_losses, 'Representation KL Loss', '#E91E63'),
    (1,0, rew_losses, 'Reward KL Loss', '#00BCD4'),
]
for r,c,data_loss,title,color in configs:
    ax = axes[r,c]
    ax.plot(train_steps, data_loss, color=color, lw=1.2)
    ax.set_title(title); ax.set_xlabel('Step'); ax.grid(alpha=0.25)

if len(image_losses) > 0:
    axes[1,1].plot(train_steps, image_losses, color='#FF9800', lw=1.2)
else:
    axes[1,1].text(0.5, 0.5, 'Image Recon Loss (data unavailable)', ha='center', va='center', fontsize=11, transform=axes[1,1].transAxes, color='gray')
axes[1,1].set_title('Image Reconstruction Loss'); axes[1,1].set_xlabel('Step'); axes[1,1].grid(alpha=0.25)

fig.suptitle('World Model Loss Curves', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'training_loss.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] training_loss.png")

# Figure 3: FPS + Random Action
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
ax1.plot(train_steps, fps_policy, color='#4CAF50', lw=1.2, label='Policy FPS')
ax1.plot(train_steps, fps_train, color='#FF9800', lw=1.2, label='Train FPS')
ax1.set_xlabel('Step'); ax1.set_ylabel('FPS')
ax1.set_title('Training Speed (RTX 4060)'); ax1.legend(); ax1.grid(alpha=0.25)

ax2.plot(train_steps, rand_actions, color='#E91E63', lw=1.5)
ax2.set_xlabel('Step'); ax2.set_ylabel('Random Action Ratio')
ax2.set_title('Exploration Ratio'); ax2.set_ylim(-0.05, 1.05); ax2.grid(alpha=0.25)
ax2.axhline(y=0.5, color='gray', ls='--', lw=0.8, alpha=0.5)
fig.suptitle('System Performance Metrics', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'training_perf.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] training_perf.png")

# Figure 4: Dashboard
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
panels = [
    (0,0, steps, scores, 'Episode Score', '#2196F3'),
    (0,1, steps, lengths, 'Episode Length', '#4CAF50'),
    (0,2, train_steps, rand_actions, 'Random Action Ratio', '#E91E63'),
    (1,0, train_steps, dyn_losses, 'Dynamic Loss', '#9C27B0'),
    (1,1, train_steps, image_losses if len(image_losses)>0 else rep_losses, 'Image Recon Loss', '#FF9800'),
    (1,2, train_steps, fps_train, 'Train FPS', '#FF5722'),
]
for r,c,x,y,title,color in panels:
    ax = axes[r,c]
    ax.plot(x, y, color=color, lw=1.2, alpha=0.9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.set_xlabel('Step', fontsize=9); ax.grid(alpha=0.2)
fig.suptitle('DreamerV3 HomeNav - Training Dashboard', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT, 'training_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"[OK] training_dashboard.png")

mask_r = steps > (max(steps) * 0.8)
print(f"\nDone. {len(os.listdir(OUT))} files in {OUT}")
print(f"  Recent Score:     {np.nanmean(scores[mask_r]):.1f}")
print(f"  Recent Length:    {np.nanmean(lengths[mask_r]):.1f}")
print(f"  Final Dyn KL:     {np.mean(dyn_losses[-20:]):.3f}")
print(f"  Final Rep KL:     {np.mean(rep_losses[-20:]):.3f}")
print(f"  Final Rew KL:     {np.mean(rew_losses[-20:]):.3f}")
print(f"  Final Rand Pct:   {rand_actions[-1]*100:.1f}%")
print(f"  Avg Train FPS:    {np.mean(fps_train):.0f}")
