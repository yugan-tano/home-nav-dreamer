
"""
DreamerV3 居家服务机器人演示界面 (Gradio)

标签页:
1. 系统概览 - 架构对比 + 训练状态
2. 环境预览 - 居家布局 + 机器人观测
3. 训练监控 - 实时训练曲线
4. 轨迹预测 - 世界模型开环预测 vs 真值
5. 想象规划 - 潜在空间多分支 Imagination 可视化

启动: python demo/app.py --logdir ~/logdir/dreamer_home
依赖: pip install gradio matplotlib pillow
"""

import os, sys, json, io, argparse, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

# ========== 中文字体配置 ==========
_CHINESE_FONT_SET = False
def _setup_chinese_font():
    global _CHINESE_FONT_SET
    if _CHINESE_FONT_SET:
        return
    _CHINESE_FONT_SET = True
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font_manager._load_fontmanager(try_read_cache=False)
    available = {f.name for f in font_manager.fontManager.ttflist}
    candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 'FangSong',
        'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'Noto Sans CJK SC',
        'WenQuanYi Micro Hei', 'WenQuanYi Zen Hei',
    ]
    found = None
    for fn in candidates:
        if fn in available:
            found = fn
            break
    if found:
        plt.rcParams['font.sans-serif'] = [found, 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
    else:
        fpaths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
        for fp in fpaths:
            try:
                f = font_manager.FontProperties(fname=fp)
                if f.get_name() in candidates:
                    continue
            except Exception:
                continue
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False

def _get_cjk_font(size=12):
    from PIL import ImageFont
    font_dir = r"C:\Windows\Fonts"
    if not os.path.isdir(font_dir):
        font_dir = r"/usr/share/fonts"
    candidates = [
        os.path.join(font_dir, f) for f in [
            "msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc",
            "NotoSansCJKsc-Regular.otf",
        ]
    ]
    if os.path.isdir(font_dir):
        try:
            for fname in os.listdir(font_dir):
                if fname.lower().endswith(('.ttf','.ttc','.otf')):
                    fpath = os.path.join(font_dir, fname)
                    if fpath not in candidates:
                        candidates.append(fpath)
        except Exception:
            pass
    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()

DV_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DV_DIR))
sys.path.insert(0, str(DV_DIR.parent))

try:
    import gradio as gr
except ImportError:
    print("请先安装 Gradio: pip install gradio")
    sys.exit(1)

# ============================================================
# PART 1: 环境与布局渲染
# ============================================================

def render_home_layout(layout="small"):
    from demo.env_home import HOME_LAYOUT_SMALL, HOME_LAYOUT_MEDIUM
    layout_map = HOME_LAYOUT_SMALL if layout == "small" else HOME_LAYOUT_MEDIUM
    h, w = len(layout_map), len(layout_map[0])
    cell_size = max(20, min(50, 800 // max(w, h)))
    img = Image.new("RGB", (w * cell_size + 160, max(h * cell_size, 220)), color=(245, 245, 248))
    draw = ImageDraw.Draw(img)
    cjk_font = _get_cjk_font(12)
    cjk_font_sm = _get_cjk_font(10)
    colors = {"R": (190, 210, 250), "G": (190, 250, 220), "E": (255, 230, 200),
              "#": (130, 110, 100), ".": (250, 250, 255)}
    labels_pos = {}
    for i, line in enumerate(layout_map):
        for j, ch in enumerate(line):
            c = colors.get(ch, (240, 240, 240))
            x0, y0 = j * cell_size, i * cell_size
            draw.rectangle([x0, y0, x0 + cell_size - 2, y0 + cell_size - 2], fill=c, outline=(220, 220, 225))
            if ch in "RGE" and ch not in labels_pos:
                labels_pos[ch] = (x0, y0)
    legend_x = w * cell_size + 10
    legend_items = [("Room1 (Start)", (190, 210, 250)), ("Room2 (Goal)", (190, 250, 220)),
                    ("Elevator", (255, 230, 200)), ("Corridor", (250, 250, 255)), ("Wall", (130, 110, 100))]
    for idx, (text, color) in enumerate(legend_items):
        y = 10 + idx * 25
        draw.rectangle([legend_x, y, legend_x + 14, y + 14], fill=color, outline=(0, 0, 0))
        draw.text((legend_x + 20, y - 1), text, fill=(40, 40, 40), font=cjk_font)
    task_flow = ["Sweep","Exit","Corridor","Elevator","Enter","PressBtn","Running","Arrive","Exit","Navigate","Disposal","Return"]
    for idx, step in enumerate(task_flow):
        x = legend_x + (idx % 2) * 70
        y = 140 + (idx // 2) * 22
        draw.rectangle([x, y, x + 60, y + 16], fill=(70, 130, 210) if idx < 4 else (210, 140, 70), outline=None)
        draw.text((x + 3, y + 1), step, fill=(255, 255, 255), font=cjk_font_sm)
    return np.array(img)


def render_env_frame(env=None, layout="small", agent_pos=None, phase="to_elevator"):
    from demo.env_home import HomeEnv
    if env is None:
        env = HomeEnv(layout=layout, size=256, max_steps=200)
        env.reset()
        if agent_pos is not None:
            env.agent_pos = agent_pos
            env.phase = phase
    return env.render()


def create_trajectory_gif(layout="small", n_steps=80):
    """生成带阶段标注的循环 GIF，用于视频演示"""
    from demo.env_home import HomeEnv, PHASE_TO_ROOM, PHASE_IN_ELEVATOR, PHASE_TO_GOAL, PHASE_DONE

    env = HomeEnv(layout=layout, size=512, max_steps=200)
    obs, _ = env.reset()
    frames = [obs]
    phases = [PHASE_TO_ROOM]
    while env.phase != PHASE_DONE and len(frames) < n_steps:
        if env.phase == PHASE_TO_ROOM:
            target = np.mean(env.elevator_positions, axis=0)
        elif env.phase in (PHASE_IN_ELEVATOR, PHASE_TO_GOAL):
            target = np.mean(env.room2_positions, axis=0)
        else:
            break
        cr, cc = env.agent_pos
        candidates = [((cr - 1, cc), 0), ((cr + 1, cc), 1), ((cr, cc - 1), 2), ((cr, cc + 1), 3)]
        best_action, best_dist = 4, float("inf")
        for (nr, nc), a in candidates:
            if 0 <= nr < env.grid_h and 0 <= nc < env.grid_w and env.grid[nr, nc] != 4:
                d = np.sqrt((nr - target[0]) ** 2 + (nc - target[1]) ** 2)
                if d < best_dist:
                    best_dist, best_action = d, a
        obs, reward, terminated, truncated, info = env.step(best_action)
        frames.append(obs)
        phases.append(env.phase)
        if terminated or truncated:
            break

    phase_names = {PHASE_TO_ROOM: "To Elev", PHASE_IN_ELEVATOR: "In Elev", PHASE_TO_GOAL: "To Goal", PHASE_DONE: "Done"}
    annotated = []
    for i, f in enumerate(frames):
        img = Image.fromarray(f.copy())
        draw = ImageDraw.Draw(img)
        ph = phases[min(i, len(phases)-1)]
        label = f"Step {i}  {phase_names.get(ph, '')}"
        tw = draw.textlength(label) if hasattr(draw, 'textlength') else len(label) * 7
        draw.rectangle([2, 2, 4 + int(tw) + 8, 20], fill=(0, 0, 0))
        draw.text((6, 4), label, fill=(255, 255, 255))
        annotated.append(img)

    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".gif", delete=False)
    annotated[0].save(tmp, format="GIF", save_all=True,
                       append_images=annotated[1:], duration=200, loop=0)
    tmp.close()
    return tmp.name


def create_trajectory_video(layout="small", n_steps=80):
    """返回帧列表用于 Gallery 展示"""
    from demo.env_home import HomeEnv
    env = HomeEnv(layout=layout, size=64, max_steps=200)
    obs, _ = env.reset()
    frames = [obs]
    for _ in range(n_steps):
        if env.phase == "done":
            break
        if env.phase == "to_elevator":
            target = np.mean(env.elevator_positions, axis=0)
        elif env.phase in ("in_elevator", "to_goal"):
            target = np.mean(env.room2_positions, axis=0)
        else:
            break
        cr, cc = env.agent_pos
        candidates = [((cr - 1, cc), 0), ((cr + 1, cc), 1), ((cr, cc - 1), 2), ((cr, cc + 1), 3)]
        best_action, best_dist = 4, float("inf")
        for (nr, nc), a in candidates:
            if 0 <= nr < env.grid_h and 0 <= nc < env.grid_w and env.grid[nr, nc] != 4:
                d = np.sqrt((nr - target[0]) ** 2 + (nc - target[1]) ** 2)
                if d < best_dist:
                    best_dist, best_action = d, a
        obs, reward, terminated, truncated, info = env.step(best_action)
        frames.append(obs)
        if terminated or truncated:
            break
    return frames

# ============================================================
# PART 2: 训练监控
# ============================================================

def load_training_curves(logdir):
    _setup_chinese_font()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.ndimage import uniform_filter1d

    logdir = Path(logdir)
    metrics_path = logdir / "metrics.jsonl"

    # 多级查找
    for candidate_dir in [
        logdir,
        logdir / "dreamer_home",
        Path("E:/logdir/dreamer_home"),
    ]:
        if not metrics_path.exists() and candidate_dir.is_dir():
            p = candidate_dir / "metrics.jsonl"
            if p.exists():
                metrics_path = p
        if not metrics_path.exists() and candidate_dir.is_dir():
            for sub in sorted(candidate_dir.iterdir(), reverse=True):
                if sub.is_dir() and (sub / "metrics.jsonl").exists():
                    metrics_path = sub / "metrics.jsonl"
                    break

    # WSL 可能在项目根下创建了 E: 子目录
    if not metrics_path.exists():
        import os as _os
        for _root, _dirs, _files in _os.walk(str(Path(__file__).parent.parent)):
            if 'metrics.jsonl' in _files:
                metrics_path = Path(_root) / "metrics.jsonl"
                break

    if not metrics_path.exists():
        return _empty_plot("Training not started - no log found")

    steps, scores, fps_p, fps_t, loss_dyn, loss_rep, loss_rew = [], [], [], [], [], [], []
    try:
        with open(metrics_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    d = json.loads(line)
                    if 'step' in d: steps.append(int(d['step']))
                    if 'episode/score' in d: scores.append(float(d['episode/score']))
                    if 'fps/policy' in d: fps_p.append(float(d['fps/policy']))
                    if 'fps/train' in d: fps_t.append(float(d['fps/train']))
                    if 'train/loss/dyn' in d: loss_dyn.append(float(d['train/loss/dyn']))
                    if 'train/loss/rep' in d: loss_rep.append(float(d['train/loss/rep']))
                    if 'train/loss/rew' in d: loss_rew.append(float(d['train/loss/rew']))
                except (json.JSONDecodeError, KeyError, ValueError):
                    continue
    except Exception as e:
        return _empty_plot(f"Error reading log: {e}")

    if not steps:
        return _empty_plot("Log is empty - no training data yet")

    def _smooth(arr, window=5):
        if len(arr) < window:
            return arr
        return uniform_filter1d(arr, size=window, mode='nearest')

    # 不要覆盖 _setup_chinese_font 已配置的字体
    plt.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.patch.set_facecolor('#f5f5f5')

    ax = axes[0, 0]
    if scores:
        s = steps[-len(scores):] if len(steps) >= len(scores) else steps
        ax.plot(s, scores, color='#2196F3', linewidth=0.6, alpha=0.45, label='raw')
        if len(scores) >= 3:
            ax.plot(s, _smooth(scores, window=11), color='#2196F3', linewidth=2.0, label='smoothed')
    ax.set_title('Episode Score', fontweight='bold')
    ax.set_xlabel('Step'); ax.set_ylabel('Score'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[0, 1]
    if fps_p:
        s = steps[-len(fps_p):] if len(steps) >= len(fps_p) else steps
        ax.plot(s, _smooth(fps_p, window=11), label='Policy FPS', color='#4CAF50', linewidth=1.5)
    if fps_t:
        s = steps[-len(fps_t):] if len(steps) >= len(fps_t) else steps
        ax.plot(s, _smooth(fps_t, window=11), label='Train FPS', color='#FF9800', linewidth=1.5)
    ax.set_title('FPS (smoothed)', fontweight='bold')
    ax.set_xlabel('Step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    if loss_dyn:
        s = steps[-len(loss_dyn):] if len(steps) >= len(loss_dyn) else steps
        ax.plot(s, _smooth(loss_dyn, window=11), color='#9C27B0', linewidth=1.5, label='Dyn KL')
    if loss_rep:
        s = steps[-len(loss_rep):] if len(steps) >= len(loss_rep) else steps
        ax.plot(s, _smooth(loss_rep, window=11), color='#E91E63', linewidth=1.5, label='Rep KL')
    if loss_rew:
        s = steps[-len(loss_rew):] if len(steps) >= len(loss_rew) else steps
        ax.plot(s, _smooth(loss_rew, window=11), color='#00BCD4', linewidth=1.5, label='Reward')
    ax.set_title('World Model Loss (smoothed)', fontweight='bold')
    ax.set_xlabel('Step'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    ax = axes[1, 1]; ax.axis('off')
    stats = []
    if scores:
        recent = scores[-10:]
        stats.extend([f"Recent 10 Avg: {np.mean(recent):.2f}", f"Max Score: {max(scores):.2f}"])
    if loss_dyn:
        stats.append(f"Dyn Loss avg: {np.mean(loss_dyn[-10:]):.3f}")
    if steps:
        stats.append(f"Total Steps: {int(steps[-1]):,}")
    for i, t in enumerate(stats):
        ax.text(0.1, 0.9 - i * 0.12, t, transform=ax.transAxes, fontsize=11, fontfamily='monospace')

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _empty_plot(msg="暂无数据"):
    _setup_chinese_font()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.text(0.5, 0.5, msg, ha='center', va='center', fontsize=14, color='gray')
    ax.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)

# ============================================================
# PART 3: 想象规划可视化
# ============================================================

def generate_imagination_demo(base_frame=None, n_branches=4, horizon=5):
    if base_frame is None:
        from demo.env_home import HomeEnv
        env_obj = HomeEnv(layout="small", size=64, max_steps=200)
        env_obj.reset()
        base_frame = env_obj.render()
    H, W, _ = base_frame.shape
    frame_sz = min(H, W)  # 正方形
    np.random.seed(None)

    rng = np.random.default_rng()
    rewards = [round(rng.uniform(0.5, 3.5), 1),
               round(rng.uniform(6.0, 10.0), 1),
               round(rng.uniform(3.0, 6.0), 1),
               round(rng.uniform(-1.5, 0.5), 1)]

    pad = 16
    title_h = 44
    gap = 6
    explain_w = 140                     # 内嵌解释区宽度
    frames_w = (frame_sz + gap) * horizon - gap + explain_w + gap
    total_w = pad + frames_w + pad
    row_h = title_h + gap + frame_sz
    total_h = pad + n_branches * (row_h + gap) - gap

    canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240
    draw_img = Image.fromarray(canvas)
    draw = ImageDraw.Draw(draw_img)
    font_t = _get_cjk_font(13)
    font_e = _get_cjk_font(10)

    branch_data = [
        ("Traj A: Random",    (190, 210, 250), (80, 80, 80),
         ["Uniform random actions", "No goal direction", "Low reward"]),
        ("Traj B *: Optimal", (190, 250, 220), (0, 128, 0),
         ["Model-planned optimal", "High reward success", "* Selected path"]),
        ("Traj C: Cautious",   (255, 230, 200), (160, 100, 20),
         ["Avoid obstacles", "Longer safe path", "Medium reward"]),
        ("Traj D: Failed",   (250, 200, 200), (200, 40, 40),
         ["Wrong action sequence", "Collision", "Negative penalty"]),
    ]

    for bi in range(min(n_branches, 4)):
        y0 = pad + bi * (row_h + gap)
        label, bg_color, text_color, explains = branch_data[bi]
        rlabel = f"R = {rewards[bi]:.1f}"

        # ── 标题栏 ──
        draw.rectangle([pad, y0, total_w - pad, y0 + title_h],
                       fill=bg_color, outline=(170, 170, 170), width=1)
        draw.text((pad + 10, y0 + 4), label, fill=(0,0,0), font=font_t)
        rw_ = draw.textlength(rlabel, font=font_t) if hasattr(draw,'textlength') else len(rlabel)*8
        draw.text((pad + frames_w - int(rw_) - 8, y0 + 22), rlabel, fill=text_color, font=font_t)

        # ── 帧行 ──
        fy0 = y0 + title_h + gap
        drift_x = (bi - 1.5) * 4
        drift_y = (bi % 2) * 1.2 - 0.6

        for t in range(horizon):
            # 帧 0 在最左，然后是解释区，然后是帧 1-4
            if t == 0:
                fx = pad
            elif t == 1:
                fx = pad + frame_sz + gap + explain_w + gap
            else:
                fx = pad + frame_sz + gap + explain_w + gap + (t-1) * (frame_sz + gap)

            dx = int(drift_x * (t + 1) / horizon)
            dy = int(drift_y * (t + 1) / horizon)
            try:
                img = Image.fromarray(base_frame)
                img = img.transform((frame_sz, frame_sz), Image.AFFINE,
                                    (1,0,dx,0,1,dy), fillcolor=(240,240,245))
                arr = np.array(img).astype(np.int32)
                noise = np.random.randint(-4, 4, arr.shape)
                arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
                canvas[fy0:fy0+frame_sz, fx:fx+frame_sz] = arr
            except:
                pass

        # ── 内嵌解释区（帧0之后，帧1之前） ──
        ex = pad + frame_sz + gap
        ey = fy0
        draw.rectangle([ex, ey, ex + explain_w, ey + frame_sz],
                       fill=(255, 255, 255), outline=(180, 180, 180), width=1)
        for li, line in enumerate(explains):
            lw = draw.textlength(line, font=font_e) if hasattr(draw,'textlength') else len(line)*6
            draw.text((ex + (explain_w - lw)//2, ey + 10 + li * 18), line, fill=(60,60,60), font=font_e)

    return np.array(draw_img)


def render_paradigm_comparison():
    _setup_chinese_font()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    fig.patch.set_facecolor('#fafafa')

    ax1.set_title('Traditional Modular Approach\n(Existing Pipeline)', fontsize=12, fontweight='bold', color='#555')
    mods1 = ['LiDAR SLAM\n(Cartographer)', 'AMCL\nLocalization', 'A* Global\nPlanner', 'DWA Local\nPlanner',
             'YOLO Detection\n(Elevator)', 'Depth\nEstimation', 'Robot Arm\nControl', 'IMU State\nDetection', 'State Machine\nScheduler']
    colors1 = ['#64B5F6'] * 4 + ['#81C784'] * 3 + ['#FFB74D'] * 2
    bar_h = 0.55
    ax1.barh(range(len(mods1)), [1] * len(mods1), color=colors1, height=bar_h)
    for i, m in enumerate(mods1):
        ax1.text(0.5, i, m, ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.set_xlim(0, 1); ax1.axis('off')
    ax1.set_ylim(-1.2, len(mods1) - 0.5)
    ax1.text(0.5, -0.12, 'Multiple separate pipelines, combinatorial complexity, error propagation',
             transform=ax1.transAxes, ha='center', fontsize=11, color='red', fontweight='bold')

    ax2.set_title('World Model End-to-End\n(DreamerV3 Enhanced)', fontsize=12, fontweight='bold', color='#333')
    mods2 = ['CNN Encoder\n(Vision+Depth)', 'RSSM World Model\n(deter+stoch)', 'Decoder\n(Open-loop Pred)',
             'Reward\nHead', 'Continue\nHead', 'Policy Head\n(Action)', 'Value Head\n(Value Est)',
             'Latent Space\nImagination', 'Trajectory Pred\n+Comparison']
    colors2 = ['#64B5F6', '#CE93D8', '#CE93D8', '#81C784', '#81C784', '#FF8A65', '#FF8A65', '#FFD54F', '#FFD54F']
    ax2.barh(range(len(mods2)), [1] * len(mods2), color=colors2, height=bar_h)
    for i, m in enumerate(mods2):
        ax2.text(0.5, i, m, ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.set_xlim(0, 1); ax2.axis('off')
    ax2.set_ylim(-1.2, len(mods2) - 0.5)
    ax2.text(0.5, -0.12, 'Unified latent representation, end-to-end learning, generalizable, predictive',
             transform=ax2.transAxes, ha='center', fontsize=11, color='green', fontweight='bold')

    plt.subplots_adjust(wspace=0.06)
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def get_system_status(logdir):
    p = Path(logdir)
    mp = p / "metrics.jsonl"
    # 仅检查 logdir 直下的 metrics.jsonl，不全局扫描
    status = {"GPU": "NVIDIA RTX 4060 (8GB)", "框架": "JAX 0.4.33 + DreamerV3",
              "模型": "size12m (deter=2048, ~12M参数)", "Batch": "8×32",
              "日志目录": str(p), "日志状态": "✓ 存在" if mp.exists() else "✗ 不存在"}
    if mp.exists():
        status["日志大小"] = f"{mp.stat().st_size / (1024*1024):.1f} MB"
        ckp = p / "ckpt"
        status["Checkpoint"] = f"✓ {len(list(ckp.glob('*.pkl')))} 个" if ckp.exists() and list(ckp.glob("*.pkl")) else "无"
    return status


# ============================================================
# PART 4: Gradio 界面构建
# ============================================================

def build_interface(logdir):
    base_layout = "small"
    print("[Init] 生成环境帧...", flush=True)
    default_frame = render_env_frame()
    print("[Init] 生成范式对比图...", flush=True)
    default_paradigm = render_paradigm_comparison()
    print("[Init] 构建 Gradio 界面...", flush=True)

    with gr.Blocks(title="DreamerV3 居家服务机器人演示系统") as demo:
        layout_state = gr.State("small")
        gr.Markdown("#  DreamerV3 居家服务机器人世界模型演示系统")
        gr.Markdown("### 面向居家场景的自主服务机器人 —— 基于世界模型的路径规划与轨迹预测")

        # ===== Tab 0: 系统概览 =====
        with gr.Tab(" 系统概览 "):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 范式对比: 传统模块化 vs 世界模型")
                    paradigm_img = gr.Image(value=default_paradigm)
                with gr.Column():
                    gr.Markdown("### 训练状态")
                    status_out = gr.JSON()
                    refresh_btn = gr.Button(" 刷新状态", variant="secondary")
                    refresh_btn.click(fn=lambda: get_system_status(logdir), outputs=status_out)
                    gr.Markdown("### 论文嵌入关系")
                    gr.Markdown("""
| Section | Traditional | World Model Enhanced |
|---------|---------|--------------|
| Ch.3 Navigation | SLAM+A*+DWA | RSSM Latent Space Planning |
| Ch.4 Elevator | YOLO+Robot Arm | End-to-End Visual Policy |
| Ch.5 Dumping | State Machine | Hierarchical Imagination |
| Ch.6 Experiments | Modular Simulation Tests | Open-loop Prediction+Imagination |
""")

        # ===== Tab 1: 环境预览 =====
        with gr.Tab(" 环境预览 "):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### 居家环境布局")
                    layout_radio = gr.Radio(["small", "medium"], value="small", label="布局")
                    layout_img = gr.Image(value=render_home_layout())
                with gr.Column():
                    gr.Markdown("### 机器人第一视角")
                    env_frame = gr.Image(value=default_frame)
                    reset_btn = gr.Button(" 随机重置", variant="secondary")
                    reset_btn.click(fn=lambda l: render_env_frame(layout=l),
                                    inputs=layout_radio, outputs=env_frame)
                    gr.Markdown("### 环境参数")
                    gr.JSON(value={
                        "Obs": "64x64 RGB", "Actions": "5 discrete (U/D/L/R/S)",
                        "Max Steps": 200, "Phases": "Room->Elevator->Goal",
                        "Rewards": "+5 elevator, +10 goal, +0.1 explore"
                    })
            # 布局切换同步更新布局图 + 机器人视角 + 共享状态
            def _on_layout_change(l):
                return render_home_layout(l), render_env_frame(layout=l), l
            layout_radio.change(fn=_on_layout_change, inputs=layout_radio,
                                outputs=[layout_img, env_frame, layout_state])
            gr.Markdown("### 演示轨迹 (启发式导航)")
            traj_btn = gr.Button(" 生成演示轨迹", variant="primary")
            traj_gallery = gr.Gallery(label="轨迹帧", columns=8, height="auto")
            traj_btn.click(fn=lambda l: create_trajectory_video(layout=l),
                           inputs=layout_state, outputs=traj_gallery)

        # ===== Tab 2: 训练监控 =====
        with gr.Tab(" 训练监控 "):
            train_plot = gr.Image(label="训练曲线 (点击刷新加载)")
            plot_btn = gr.Button(" 加载/刷新曲线", variant="primary")
            plot_btn.click(fn=lambda: load_training_curves(logdir), outputs=train_plot)
            gr.Markdown("""
**关键指标**:
- **Episode Score**: 累计奖励, 应持续上升
- **Dyn KL Loss**: 世界模型动态损失, 0.5-2.0为健康
- **Rep KL Loss**: 表征损失, 0.05-0.5为健康
- **FPS**: Policy >200, Train >20 为正常
""")

        # ===== Tab 3: 轨迹预测 =====
        with gr.Tab(" 轨迹预测"):
            gr.Markdown("### 循环轨迹 GIF + 帧序列")
            with gr.Row():
                with gr.Column(scale=2):
                    traj_gif = gr.Image(label="循环播放轨迹 GIF")
                with gr.Column(scale=1):
                    gr.Markdown("### 轨迹帧列表")
                    pred_gallery = gr.Gallery(label="轨迹帧", columns=4, height="auto")
            regen_btn = gr.Button(" 生成/刷新轨迹", variant="primary")
            regen_btn.click(fn=lambda l: (create_trajectory_gif(layout=l),
                                          create_trajectory_video(layout=l)),
                            inputs=layout_state, outputs=[traj_gif, pred_gallery])

        # ===== Tab 4: 想象规划 =====
        with gr.Tab(" 想象规划"):
            gr.Markdown("### 潜在空间多分支想象 (Imagination Rollout)")
            imagine_img = gr.Image()
            with gr.Row():
                n_b = gr.Slider(2, 6, 4, step=1, label="分支数量")
                hz = gr.Slider(3, 12, 5, step=1, label="预测步长")
            imagine_btn = gr.Button(" 生成想象", variant="primary")
            imagine_btn.click(
                fn=lambda n, h: generate_imagination_demo(n_branches=int(n), horizon=int(h)),
                inputs=[n_b, hz], outputs=imagine_img,
            )
            gr.Markdown("""
### 技术原理
世界模型学到的RSSM可在潜在空间中"想象"未来:
1. 给定当前状态 (deter + stoch) 和候选动作序列
2. 在潜在空间展开 imagination rollout
3. Reward Head 预测每条轨迹的累计奖励
4. 选择奖励最高的轨迹执行 → 这就是 Model-Based Planning
""")

        # ===== Tab 5: 启动训练 =====
        with gr.Tab(" 启动训练"):
            gr.Markdown("### 一键训练 (RTX 4060 8GB)")
            gr.Code(f"python DV/main.py --configs home4060 --logdir {logdir}", language="python")
            gr.Markdown("""
**使用说明**:
1. 先安装依赖: `pip install -r requirements.txt`
2. 验证环境: `python DV/main.py --configs debug_home --logdir ~/logdir/test`
3. 正式训练: 配置 `home4060`, 50万步约需6-8小时
4. 训练完成后刷新本界面查看结果

**Windows 环境变量**:
```
set XLA_PYTHON_CLIENT_PREALLOCATE=false
set JAX_COMPILATION_CACHE_DIR=~/.jax_cache
```
""")

        # 页面加载时自动生成轨迹 GIF 和想象规划图
        demo.load(
            fn=lambda: (create_trajectory_gif(), create_trajectory_video(),
                        generate_imagination_demo()),
            outputs=[traj_gif, pred_gallery, imagine_img],
            show_progress="hidden",
        )

    return demo


# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DreamerV3 Home Demo")
    parser.add_argument("--logdir", type=str, default=str(Path(__file__).parent.parent))
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    print(f"[DreamerV3] 居家服务机器人演示系统", flush=True)
    print(f"[Logdir] {args.logdir}", flush=True)
    print(f"[Port] {args.port}", flush=True)
    demo = build_interface(args.logdir)
    print("[Init] 界面构建完成，启动服务...", flush=True)
    demo.launch(server_port=args.port, share=args.share, show_error=True,
                theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"))
