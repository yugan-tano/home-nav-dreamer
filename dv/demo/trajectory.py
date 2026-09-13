"""
轨迹预测与想象规划可视化模块

基于 DreamerV3 世界模型:
1. 加载训练好的 checkpoint
2. 将观测编码为潜在状态
3. 在潜在空间中进行 imagination rollout
4. 解码回观测空间并可视化

用于演示:
- 开环预测 (open-loop): 给定固定动作序列, 预测未来观测
- 闭环预测 (closed-loop): 策略在潜在空间中自主规划
- 轨迹对比: 真实轨迹 vs 世界模型预测轨迹
"""
import os
import sys
import json
import gzip
import pickle
import io
from pathlib import Path
from functools import partial as bind

import numpy as np
from PIL import Image, ImageDraw


class TrajectoryPredictor:
    """基于训练好的世界模型进行轨迹预测"""

    def __init__(self, logdir):
        self.logdir = Path(logdir)
        self._loaded = False
        self.agent = None
        self.config = None

    def load(self):
        """加载训练好的模型 checkpoint"""
        import jax
        print(f"Loading checkpoint from {self.logdir}...")
        print(f"Available devices: {jax.devices()}")

        # 查找 checkpoint
        ckpt_dir = self.logdir / "ckpt"
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_dir}")

        ckpt_files = sorted(ckpt_dir.glob("*.pkl"))
        if not ckpt_files:
            raise FileNotFoundError(f"No checkpoint files in {ckpt_dir}")

        latest_ckpt = ckpt_files[-1]
        print(f"Using checkpoint: {latest_ckpt}")

        with open(latest_ckpt, 'rb') as f:
            ckpt = pickle.load(f)

        self._loaded = True
        print("Checkpoint loaded successfully")
        return ckpt

    def predict_trajectory(self, obs_history, actions, horizon=30):
        """给定观测历史和动作序列, 预测未来轨迹"""
        if not self._loaded:
            self.load()

        # 这里需要访问训练好的 world model 权重
        # 实际实现需要加载完整的 Agent 并调用其方法
        raise NotImplementedError(
            "Full trajectory prediction requires trained model. "
            "Call load() first, then use agent.policy() and agent.report() methods."
        )

    @staticmethod
    def load_metrics(logdir):
        """从 JSONL 日志中加载训练指标"""
        metrics_path = Path(logdir) / "metrics.jsonl"
        if not metrics_path.exists():
            return {}

        metrics = []
        with open(metrics_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    metrics.append(json.loads(line))
        return metrics

    @staticmethod
    def load_scores(logdir):
        """从 scores.jsonl 中加载 episode 分数"""
        scores_path = Path(logdir) / "scores.jsonl"
        if not scores_path.exists():
            return []

        scores = []
        with open(scores_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        scores.append(float(data.get('episode/score', 0)))
                    except (json.JSONDecodeError, ValueError):
                        continue
        return scores


def render_predicted_frames(true_images, pred_images, error_images=None, border_size=2):
    """将真实图像、预测图像和误差图像拼接为可视化网格"""
    T = len(true_images)
    if error_images is None:
        error_images = []
        for t_img, p_img in zip(true_images, pred_images):
            error = np.abs(t_img.astype(np.float32) - p_img.astype(np.float32))
            error = (error / max(error.max(), 1.0) * 255).astype(np.uint8)
            error_images.append(error)

    frames = []
    for t in range(T):
        true_frame = Image.fromarray(true_images[t])
        pred_frame = Image.fromarray(pred_images[t])
        error_frame = Image.fromarray(error_images[t])

        h = true_frame.height
        w = true_frame.width
        combined = Image.new("RGB", (w * 3 + border_size * 4, h + border_size * 2))
        combined.paste(true_frame, (border_size, border_size))
        combined.paste(pred_frame, (w + border_size * 2, border_size))
        combined.paste(error_frame, (w * 2 + border_size * 3, border_size))

        # 添加标签条
        draw = ImageDraw.Draw(combined)
        draw.rectangle([border_size, 0, w + border_size, border_size], fill=(0, 255, 0))
        draw.rectangle([w + border_size * 2, 0, w * 2 + border_size * 2, border_size], fill=(255, 0, 0))
        draw.rectangle([w * 2 + border_size * 3, 0, w * 3 + border_size * 3, border_size], fill=(120, 120, 120))

        frames.append(combined)

    return frames


def render_trajectory_overlay(base_image, true_traj, pred_traj=None, size=256):
    """在基础地图上叠加轨迹"""
    img = Image.fromarray(base_image).convert("RGB").resize((size, size))
    # 简化实现: 直接返回原图加标记
    draw = ImageDraw.Draw(img)

    if true_traj and len(true_traj) >= 2:
        for i in range(1, len(true_traj)):
            p1 = true_traj[i - 1]
            p2 = true_traj[i]
            scale = size / max(base_image.shape[0], base_image.shape[1])
            x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
            x2, y2 = int(p2[0] * scale), int(p2[1] * scale)
            draw.line([(x1, y1), (x2, y2)], fill=(0, 255, 0), width=2)

    if pred_traj and len(pred_traj) >= 2:
        for i in range(1, len(pred_traj)):
            p1 = pred_traj[i - 1]
            p2 = pred_traj[i]
            scale = size / max(base_image.shape[0], base_image.shape[1])
            x1, y1 = int(p1[0] * scale), int(p1[1] * scale)
            x2, y2 = int(p2[0] * scale), int(p2[1] * scale)
            draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=2)

    return np.array(img)


def create_trajectory_gif(frames, output_path, duration=100):
    """从帧列表创建 GIF"""
    if not frames:
        return
    pil_frames = []
    for f in frames:
        if isinstance(f, np.ndarray):
            pil_frames.append(Image.fromarray(f))
        else:
            pil_frames.append(f)
    pil_frames[0].save(
        output_path, save_all=True, append_images=pil_frames[1:],
        duration=duration, loop=0
    )
    return output_path
