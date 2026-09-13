"""
从 HomeEnv 生成轨迹预测 GIF 动画（循环播放移动效果）
输出到 DV/slides/trajectory.gif

用法:
    python DV/scripts/gen_trajectory_gif.py
"""
import sys
from pathlib import Path

DV_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(DV_DIR))
sys.path.insert(0, str(DV_DIR.parent))

from demo.env_home import HomeEnv
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── 生成轨迹帧 ──
env = HomeEnv(layout="small", size=128, max_steps=200)
obs, _ = env.reset()
frames = [obs]
while env.phase != "done" and len(frames) < 100:
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

# ── 每帧加阶段标注 + 延迟 ──
phase_names = {"to_elevator": "→电梯", "in_elevator": "电梯内", "to_goal": "→目标", "done": "完成"}
annotated = []
for i, f in enumerate(frames * 2):  # 循环两遍
    img = Image.fromarray(f.copy())
    draw = ImageDraw.Draw(img)
    phase = env.phase if i < len(frames) else "done"
    label = f"Step {i % len(frames)}  {phase_names.get(phase, '')}"
    draw.rectangle([2, 2, len(label) * 7 + 10, 22], fill=(0, 0, 0, 128))
    draw.text((6, 4), label, fill=(255, 255, 255))
    annotated.append(img)

out = Path(__file__).parent.parent / "slides" / "trajectory.gif"
annotated[0].save(
    str(out), save_all=True, append_images=annotated[1:],
    duration=180, loop=0,
)
print(f"[OK] trajectory.gif  {len(annotated)} frames  →  {out}")
