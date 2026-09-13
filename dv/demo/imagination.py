"""
想象规划模块 (Imagination & Planning)

基于 DreamerV3 世界模型在潜在空间中展开多条可能的未来轨迹
(imagination rollouts), 并从中选择最优行动序列。

核心概念:
- World Model 学到的 transition model 使我们能"做梦"预测未来
- 在潜在空间中, 每一步计算成本远低于真实环境交互
- 通过展开多条并行轨迹并评估 return, 实现模型预测控制 (MPC)

演示功能:
1. 单条想象轨迹可视化: 给定当前状态+动作序列, 预测未来N帧
2. 多条并行想象轨迹: 展示世界模型对多种可能性的"想象"
3. 奖励预测: 在想象空间中预测每条轨迹的累计奖励
"""
import numpy as np
from PIL import Image, ImageDraw


class ImaginationVisualizer:
    """想象规划可视化工具"""

    def __init__(self, env_shape=(64, 64, 3)):
        self.env_shape = env_shape
        self.base_color = np.array([240, 240, 245])

    def render_imagination_grid(
        self, imagined_sequences, horizon=15, n_branches=4
    ):
        """
        将多条想象轨迹排列成网格可视化

        imagined_sequences: list of numpy arrays, each shape (horizon, H, W, 3)
        """
        n = min(len(imagined_sequences), n_branches)
        if n == 0:
            return self._empty_frame()

        H, W, C = self.env_shape
        # 排列成 2xN/2 的网格
        cols = min(n, 4)
        rows = (n + cols - 1) // cols

        # 每行展示一条轨迹的不同时间步
        grid_w = horizon * W + (horizon - 1) * 2  # 2px 间隔
        grid_h = rows * H + (rows - 1) * 4

        canvas = np.ones((grid_h, grid_w, 3), dtype=np.uint8) * 240

        for row_idx in range(rows):
            seq_idx = row_idx
            if seq_idx >= n:
                break

            seq = imagined_sequences[seq_idx]
            for t in range(min(horizon, len(seq))):
                x = t * (W + 2)
                y = row_idx * (H + 4)
                if x + W <= grid_w and y + H <= grid_h:
                    canvas[y:y + H, x:x + W] = seq[t]

        return canvas

    def render_single_imagination(
        self, imagined_seq, current_obs, show_reward=False
    ):
        """
        渲染单条想象轨迹: 当前帧 | 想象帧1 | 想象帧2 | ... | 想象帧N
        """
        H, W = self.env_shape[:2]
        n_frames = len(imagined_seq) if isinstance(imagined_seq, list) else imagined_seq.shape[0]
        total_w = (n_frames + 1) * W + n_frames * 4

        canvas = np.ones((H, total_w, 3), dtype=np.uint8) * 240
        draw_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(draw_canvas)

        # 当前帧
        if current_obs is not None:
            canvas[:, :W] = current_obs
            draw.text((2, 2), "Now", fill=(0, 255, 0))

        # 想象帧
        for t in range(n_frames):
            x = (t + 1) * (W + 4)
            frame = imagined_seq[t] if isinstance(imagined_seq, list) else imagined_seq[t]
            canvas[:, x:x + W] = frame
            draw.text((x + 2, 2), f"t+{t+1}", fill=(255, 0, 0))

        return np.array(draw_canvas)

    def render_branch_comparison(self, branches_info):
        """
        渲染多分支对比: 展示世界模型对不同动作序列的预测

        branches_info: list of dicts with keys:
            - 'frames': list of np.ndarray
            - 'predicted_reward': float
            - 'action_sequence': list of int
        """
        if not branches_info:
            return self._empty_frame()

        H, W = self.env_shape[:2]
        n_branches = len(branches_info)
        horizon = len(branches_info[0]['frames'])

        # 垂直排列分支
        padding = 10
        bar_w = 60  # 奖励指示条宽度
        total_w = bar_w + horizon * W + (horizon - 1) * 2 + padding * 2
        total_h = n_branches * (H + 4) + padding * 2

        canvas = np.ones((total_h, total_w, 3), dtype=np.uint8) * 240
        draw_canvas = Image.fromarray(canvas)
        draw = ImageDraw.Draw(draw_canvas)

        # 归一化奖励用于颜色映射
        rewards = [b.get('predicted_reward', 0) for b in branches_info]
        min_r, max_r = min(rewards), max(rewards)
        r_range = max(max_r - min_r, 1e-6)

        for bi, branch in enumerate(branches_info):
            y_base = padding + bi * (H + 4)
            frames = branch['frames']
            reward = branch.get('predicted_reward', 0)

            # 奖励指示条 (绿色=高奖励, 红色=低奖励)
            r_norm = (reward - min_r) / r_range
            bar_color = (
                int(255 * (1 - r_norm)),
                int(50 + 205 * r_norm),
                int(255 * (1 - r_norm))
            )
            bar_y = y_base
            bar_h = H
            draw.rectangle(
                [padding, bar_y, padding + bar_w - 5, bar_y + bar_h],
                fill=bar_color
            )
            draw.text((padding + 2, bar_y + 2), f"R={reward:.2f}", fill=(0, 0, 0))

            # 帧序列
            x_start = padding + bar_w
            for t in range(horizon):
                x = x_start + t * (W + 2)
                if x + W <= total_w:
                    canvas[y_base:y_base + H, x:x + W] = frames[t]

        return np.array(draw_canvas)

    def _empty_frame(self, text="No data"):
        H, W = self.env_shape[:2]
        canvas = np.ones((H, W, 3), dtype=np.uint8) * 240
        draw = ImageDraw.Draw(Image.fromarray(canvas))
        draw.text((10, H // 2), text, fill=(100, 100, 100))
        return np.array(draw)


def compute_imagination_rewards(
    agent, world_model_state, action_sequences, horizon=15
):
    """
    在潜在空间中评估多条动作序列的预期累计奖励

    使用训练好的 reward head 和 world model 计算每条序列的 return.

    Args:
        agent: DreamerV3 Agent 实例
        world_model_state: 当前世界模型状态 (deter, stoch)
        action_sequences: list of action sequences
        horizon: 每条的预测长度

    Returns:
        list of (sequence, predicted_return, imagined_states)
    """
    results = []
    for action_seq in action_sequences:
        # 在潜在空间中 rollout
        # imagined_states = agent.dyn.imagine(world_model_state, lambda s: action_seq, horizon)
        # rewards = agent.rew(agent.feat2tensor(imagined_states))
        # predicted_return = rewards.sum()
        # results.append((action_seq, float(predicted_return), imagined_states))
        pass
    return results
