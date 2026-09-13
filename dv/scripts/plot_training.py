"""
从训练日志 metrics.jsonl 生成训练曲线图 (自动过滤稀疏 NaN 行)
用法:
    python DV/scripts/plot_training.py --logdir /path --out DV/slides/
"""

import json, argparse, sys, io
from pathlib import Path
import numpy as np


def read_metrics(logdir):
    path = Path(logdir) / "metrics.jsonl"
    if not path.exists():
        candidates = sorted(Path(logdir).glob("*/metrics.jsonl"))
        path = candidates[-1] if candidates else None
        if not path:
            print(f"[ERROR] metrics.jsonl not found: {logdir}")
            sys.exit(1)
        print(f"[INFO] auto-detected: {path}")

    # sparse metrics: only in report lines. store (step, val) pairs
    sparse = {
        "fps_policy": [], "fps_train": [],
        "loss_dyn": [], "loss_image": [], "loss_value": [], "loss_rew": [],
        "rand_action": [],
    }
    # dense metrics: in every line
    steps, scores, episode_len = [], [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            s = d.get("step")
            if s is None:
                continue
            st = int(s)
            steps.append(st)
            scores.append(float(d.get("episode/score", float("nan"))))
            episode_len.append(float(d.get("episode/length", float("nan"))))

            for key, bucket in sparse.items():
                k = {"fps_policy": "fps/policy", "fps_train": "fps/train",
                     "loss_dyn": "train/loss/dyn", "loss_image": "train/loss/image",
                     "loss_value": "train/loss/value", "loss_rew": "train/loss/rew",
                     "rand_action": "train/rand/action"}[key]
                v = d.get(k)
                if v is not None:
                    bucket.append((st, float(v)))

    # convert dense lists to numpy
    result = {
        "steps": np.array(steps, dtype=np.float64),
        "scores": np.array(scores, dtype=np.float64),
        "episode_len": np.array(episode_len, dtype=np.float64),
    }
    for key, bucket in sparse.items():
        if bucket:
            xs, ys = zip(*bucket)
            result[key] = (np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64))
        else:
            result[key] = (np.array([], dtype=np.float64), np.array([], dtype=np.float64))
    return result


def _filter_nan(x, y):
    """plot helper: remove NaN before plotting"""
    mask = np.isfinite(y)
    return x[mask], y[mask]


def plot_curves(data, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager as _fm
    _fm._load_fontmanager(try_read_cache=False)
    available = {f.name for f in _fm.fontManager.ttflist}
    candidates_zh = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'Noto Sans CJK SC']
    font_list = ['DejaVu Sans']
    for fn in candidates_zh:
        if fn in available:
            font_list = [fn, 'DejaVu Sans']
            break
    plt.rcParams["font.sans-serif"] = font_list
    plt.rcParams["axes.unicode_minus"] = False

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    s = data["steps"]

    # ── Figure 1: Score + Episode Length ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    sx, sy = _filter_nan(s, data["scores"])
    ax1.plot(sx, sy, color="#2196F3", linewidth=1.5, alpha=0.9)
    ax1.set_xlabel("Steps", fontsize=12)
    ax1.set_ylabel("Episode Score", fontsize=12)
    ax1.set_title("Episode Score", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.25)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=0.8, alpha=0.5)
    if len(sy) > 100:
        window = max(len(sy) // 40, 10)
        ma = np.convolve(sy, np.ones(window) / window, mode="valid")
        ax1.plot(sx[window - 1:], ma, color="#FF5722", linewidth=2, label=f"MA({window})")
        ax1.legend(fontsize=10)

    lx, ly = _filter_nan(s, data["episode_len"])
    ax2.plot(lx, ly, color="#4CAF50", linewidth=1.2)
    ax2.set_xlabel("Steps", fontsize=12)
    ax2.set_ylabel("Episode Length", fontsize=12)
    ax2.set_title("Episode Length", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.25)

    fig.suptitle(f"DreamerV3 HomeNav Training Progress  ({int(s[-1])} steps)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_score.png", dpi=150, bbox_inches="tight")
    plt.close()
    n_fps = len(data.get("fps_train", ([], []))[0])
    print(f"[OK] training_score.png  ({len(sx)} score points, {n_fps} report points)")

    # ── Figure 2: Loss curves (sparse) ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    config = [
        (0, 0, "loss_dyn", "Dynamic Loss", "#9C27B0"),
        (0, 1, "loss_image", "Image Reconstruction Loss", "#FF9800"),
        (1, 0, "loss_value", "Value Loss", "#4CAF50"),
        (1, 1, "loss_rew", "Reward Loss", "#2196F3"),
    ]
    for r, c, key, title, color in config:
        ax = axes[r, c]
        xs, ys = data[key]
        if len(xs) > 0:
            ax.plot(xs, ys, color=color, linewidth=1.2, alpha=0.85)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Steps", fontsize=11)
        ax.grid(True, alpha=0.25)
    fig.suptitle("World Model Loss Curves", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] training_loss.png")

    # ── Figure 3: FPS + Random Action (sparse) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    for key, label, color in [("fps_policy", "Policy FPS", "#4CAF50"), ("fps_train", "Train FPS", "#FF9800")]:
        xs, ys = data[key]
        if len(xs) > 0:
            ax1.plot(xs, ys, color=color, linewidth=1.2, label=label)
    ax1.set_xlabel("Steps", fontsize=12); ax1.set_ylabel("FPS", fontsize=12)
    ax1.set_title("Training Speed (RTX 4060)", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10); ax1.grid(True, alpha=0.25)

    rx, ry = data["rand_action"]
    if len(rx) > 0:
        ax2.plot(rx, ry, color="#E91E63", linewidth=1.5)
    ax2.set_xlabel("Steps", fontsize=12); ax2.set_ylabel("Random Action Ratio", fontsize=12)
    ax2.set_title("Exploration vs Policy", fontsize=14, fontweight="bold")
    ax2.set_ylim(-0.05, 1.05); ax2.grid(True, alpha=0.25)
    ax2.axhline(y=0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5, label="50%")
    ax2.legend(fontsize=10)

    fig.suptitle("System Performance Metrics", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_perf.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] training_perf.png")

    # ── Figure 4: Dashboard ──
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    dash = [
        (0, 0, "scores", None, "Episode Score", "#2196F3"),
        (0, 1, "episode_len", None, "Episode Length", "#4CAF50"),
        (0, 2, "rand_action", None, "Random Action Ratio", "#E91E63"),
        (1, 0, "loss_dyn", None, "Dynamic Loss", "#9C27B0"),
        (1, 1, "loss_image", None, "Image Recon Loss", "#FF9800"),
        (1, 2, "fps_train", None, "Train FPS", "#FF5722"),
    ]
    for r, c, key, _, title, color in dash:
        ax = axes[r, c]
        if data.get(key) is not None:
            xs, ys = (s, data[key]) if isinstance(data[key], np.ndarray) else data[key]
            xs, ys = _filter_nan(xs, ys)
            if len(xs) > 0:
                ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.9)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Steps", fontsize=10)
        ax.grid(True, alpha=0.2)
    fig.suptitle("DreamerV3 HomeNav - Training Dashboard", fontsize=16, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "training_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("[OK] training_dashboard.png")
    print(f"\nAll plots saved to: {out_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", default="/mnt/e/logdir/dreamer_home")
    parser.add_argument("--out", default="DV/slides/")
    args = parser.parse_args()
    data = read_metrics(args.logdir)
    n_report = len(data.get("fps_train", ([], []))[0])
    score_vals = data["scores"][np.isfinite(data["scores"])] if len(data["scores"]) > 0 else np.array([0])
    print(f"Loaded {len(data['steps'])} records ({n_report} report lines)")
    print(f"  Steps: {int(data['steps'][0])} -> {int(data['steps'][-1])}")
    print(f"  Score: {score_vals.min():.1f} -> {score_vals.max():.1f}")
    plot_curves(data, args.out)


if __name__ == "__main__":
    main()
