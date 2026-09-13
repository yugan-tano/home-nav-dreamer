# 代码说明文档

## 项目简介

基于 DreamerV3 世界模型的居家服务机器人导航系统。本项目构建了 HomeNav-v0 仿真环境，模拟住宅场景中服务机器人从房间出发、经由走廊到达电梯、最终抵达目标房间的导航任务链。使用 RSSM（Recurrent State Space Model）世界模型在潜在空间中进行想象规划，实现端到端的视觉导航决策。

## 文件结构

```
code/
├── env_home.py              # 居家导航仿真环境 (HomeNav-v0)
├── config_home4060.yaml     # RTX 4060 训练超参数配置
├── agent.py                 # DreamerV3 Agent 完整实现
├── rssm.py                  # RSSM 世界模型核心实现
├── main.py                  # 训练/评估主入口
├── imagination.py           # 想象规划可视化模块
├── trajectory.py            # 轨迹预测模块
├── plot_training.py         # 训练曲线生成脚本
├── gen_figures.py           # 论文图表生成脚本
├── gen_demo_figures.py      # 演示图表生成（架构图、布局图等）
├── generate_report.py       # 完整训练报告生成
├── requirements.txt         # Python 依赖清单
└── README.md                # 本文件
```

## 环境要求

- Python 3.11+
- JAX 0.4.33+ (CUDA 12)
- CUDA Toolkit 12.x
- NVIDIA GPU (推荐 RTX 4060 8GB 或更高)
- Gradio 4.x (演示界面)
- NumPy, Matplotlib, Pillow, SciPy, PyYAML

关键依赖安装：

```bash
pip install jax[cuda12] jaxlib gymnasium numpy pillow matplotlib scipy pyyaml gradio einops
```

## 运行命令

### 环境测试

```bash
python env_home.py
```

### 正式训练 (RTX 4060, 500,000 steps)

```bash
# Windows 环境变量
set XLA_PYTHON_CLIENT_PREALLOCATE=false

python main.py --configs home4060 --logdir ../data/
```

### 生成训练曲线

```bash
python plot_training.py --logdir ../data/ --out ../paper/figures/
```

### 生成论文图表

```bash
# 训练曲线图表（需要 metrics.jsonl）
python gen_figures.py

# 演示图表（架构图、布局图、想象规划图）
python gen_demo_figures.py
```

### 生成训练报告

```bash
python generate_report.py --logdir ../data/ --out ../paper/figures/
```

## 关键技术参数

| 参数 | 值 |
|------|-----|
| 模型参数总量 | ~10,492,103 |
| RSSM 确定性维度 | 2048 |
| RSSM 随机状态 | 16 组 × 16 类别 |
| 观测分辨率 | 64 × 64 RGB |
| 动作空间 | 5 离散动作 (上下左右停) |
| 训练步数 | 500,000 |
| 批大小 | 8 × 32 |
| 想象规划长度 | 15 步 |
| Replay Buffer | 500,000 transitions |
| 训练时长 (RTX 4060) | ~6-8 小时 |

## 依赖的 DV 框架

本项目核心算法实现依赖于 DreamerV3 的 DV 框架。该框架包含 `embodied` 核心库和 `DV` 算法实现，位于原始项目的 `DV/` 目录中。如需完整运行训练，请将 `DV/` 目录复制到代码同级目录下，或将 `DV/` 添加到 Python 路径中。

## 论文引用

如果本项目对您的研究有帮助，请引用：

- Hafner D, et al. "Mastering Diverse Domains through World Models." arXiv:2301.04104, 2023.
