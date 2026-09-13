# 基于 DreamerV3 世界模型的居家服务机器人导航系统

# Home Service Robot Navigation with DreamerV3 World Models

[English](#english) · [中文](#chinese)

---

<a id="english"></a>
## English

**Home Service Robot Navigation with DreamerV3 World Models** is an end-to-end
visual navigation system built on the DreamerV3 world model. Instead of the
traditional "perception → mapping → planning → control" modular pipeline, it
uses a Recurrent State Space Model (RSSM) as a unified hub to jointly learn
environment representations, dynamics, and rewards, and performs multi-branch
imagination planning in the latent space for decision making.

### Key Highlights

- **World-model-based end-to-end navigation**: replaces the error-accumulating
  modular architecture with an RSSM-centered unified model.
- **Multi-branch imagination planning**: predicts and ranks candidate action
  sequences in latent space without interacting with the environment.
- **Self-built HomeNav-v0 environment**: a Gymnasium-compliant grid-world
  simulating a home service robot navigating `Room1 → Corridor → Elevator →
  Corridor → Room2`.
- **Layered composite reward**: phased goal rewards, exploration bonus,
  distance shaping, and movement penalty.
- **Full visualization demo**: a Gradio-based interactive system for training
  curves, imagination trajectories, and environment layout.

### Repository Structure

```
├── paper/                  # LaTeX source of the paper
├── code/                   # Core implementation (modified DreamerV3 + env)
├── data/                   # Training logs and sample trajectories
├── dv/                     # Full DreamerV3 framework (incl. embodied, demo)
├── LICENSE
└── README.md
```

### Results

After 500,000 environment steps of training on an RTX 4060 (8 GB):

| Metric | Initial | Final |
|---|---|---|
| Episode Score | ≈ -10 | ≈ +15 |
| Episode Length | ≈ 200 steps | ≈ 20 steps (≈10× faster) |
| Dynamics KL | 4.36 nats | 3.07 nats |
| Image reconstruction loss | 176 | 35 |
| Random action ratio | 99% | 9% |
| Success rate | < 0.1% | > 90% |

### Setup

```bash
pip install jax[cuda12]==0.4.33 optax "numpy<2" scipy gymnasium Pillow \
    PyYAML elements ninjax portal scope einops tqdm chex jaxtyping \
    gradio matplotlib nvidia-cuda-nvcc-cu12<=12.2
```

The full DreamerV3 framework (including the `embodied` and `elements`
dependencies) is bundled under `dv/`. Add it to `PYTHONPATH` or install it:

```bash
pip install -e dv/
```

### Usage

```bash
# Environment smoke test
python code/env_home.py

# Train (500k steps)
set XLA_PYTHON_CLIENT_PREALLOCATE=false
python code/main.py --configs home4060 --logdir ./logs

# Plot training curves
python code/plot_training.py --logdir ./data --out ./paper/figures

# Launch the Gradio demo
python dv/demo/app.py
```

### Dataset & Model Weights

Training logs (`data/metrics.jsonl`, `data/scores.jsonl`) and sample
trajectories (`data/sample_trajectories.json`) are included in this repository.
The trained model checkpoint (`agent.pkl`, ~123 MB) is published as a
[GitHub Release](https://github.com/REPO_OWNER/REPO_NAME/releases) asset due to
its size. You can also reproduce the weights by re-running training.

### Paper

The LaTeX source is in `paper/paper.tex`. Compile with XeLaTeX + CTeX.

### Citation

```bibtex
@misc{chen2026homenav,
  author       = {Shanshan Chen},
  title        = {Home Service Robot Navigation with DreamerV3 World Models},
  year         = {2026},
  howpublished = {\url{https://github.com/REPO_OWNER/REPO_NAME}},
}
```

Based on:

- Hafner D., Pasukonis J., Ba J., Lillicrap T. *Mastering Diverse Domains
  through World Models.* arXiv:2301.04104, 2023.

---

<a id="chinese"></a>
## 中文

**基于 DreamerV3 世界模型的居家服务机器人导航系统**是一套端到端视觉导航系统。
与传统"感知 → 建图 → 规划 → 控制"的模块化架构不同，本系统以 RSSM（循环状态
空间模型）为统一信息中枢，联合学习环境表征、动力学与奖励，并在潜在空间中执行
多分支想象规划完成决策。

### 核心亮点

- **基于世界模型的端到端导航**：以 RSSM 为核心的统一模型取代误差逐级累积的模块化架构。
- **多分支想象规划**：无需与环境交互，即可在潜在空间中对候选动作序列进行前瞻推演并择优执行。
- **自建 HomeNav-v0 环境**：符合 Gymnasium 规范的网格世界，模拟居家服务机器人
  `房间 1 → 走廊 → 电梯 → 走廊 → 房间 2` 的导航任务链。
- **分层复合奖励**：阶段性目标奖励、探索激励、距离引导、移动惩罚四层设计。
- **全流程可视化演示**：基于 Gradio 的交互式系统，展示训练曲线、想象轨迹与环境布局。

### 目录结构

```
├── paper/                  # 论文 LaTeX 源文件
├── code/                   # 核心实现（修改版 DreamerV3 + 环境）
├── data/                   # 训练日志与示例轨迹
├── dv/                     # 完整 DreamerV3 框架（含 embodied、demo）
├── LICENSE
└── README.md
```

### 实验结果

在 RTX 4060（8 GB）上完成 500,000 环境步训练后：

| 指标 | 初始值 | 最终值 |
|---|---|---|
| Episode Score | ≈ -10 | ≈ +15 |
| Episode Length | ≈ 200 步 | ≈ 20 步（效率约 10 倍） |
| 动力学 KL 散度 | 4.36 nats | 3.07 nats |
| 图像重建损失 | 176 | 35 |
| 随机动作比例 | 99% | 9% |
| 导航成功率 | < 0.1% | > 90% |

### 环境配置

```bash
pip install jax[cuda12]==0.4.33 optax "numpy<2" scipy gymnasium Pillow \
    PyYAML elements ninjax portal scope einops tqdm chex jaxtyping \
    gradio matplotlib nvidia-cuda-nvcc-cu12<=12.2
```

完整的 DreamerV3 框架（含 `embodied`、`elements` 等依赖）已内置于 `dv/` 目录，
可将其加入 `PYTHONPATH` 或直接安装：

```bash
pip install -e dv/
```

### 运行方式

```bash
# 环境冒烟测试
python code/env_home.py

# 正式训练（50 万步）
set XLA_PYTHON_CLIENT_PREALLOCATE=false
python code/main.py --configs home4060 --logdir ./logs

# 绘制训练曲线
python code/plot_training.py --logdir ./data --out ./paper/figures

# 启动 Gradio 演示
python dv/demo/app.py
```

### 数据集与模型权重

本仓库已包含训练日志（`data/metrics.jsonl`、`data/scores.jsonl`）与示例轨迹
（`data/sample_trajectories.json`）。训练好的模型权重（`agent.pkl`，约 123 MB）
因体积较大，以 [GitHub Release](https://github.com/REPO_OWNER/REPO_NAME/releases)
资产的形式发布；也可通过重新运行训练复现。

### 论文

LaTeX 源文件位于 `paper/paper.tex`，使用 XeLaTeX + CTeX 编译。

### 引用

```bibtex
@misc{chen2026homenav,
  author       = {Shanshan Chen},
  title        = {Home Service Robot Navigation with DreamerV3 World Models},
  year         = {2026},
  howpublished = {\url{https://github.com/REPO_OWNER/REPO_NAME}},
}
```

本项目基于：

- Hafner D., Pasukonis J., Ba J., Lillicrap T. *Mastering Diverse Domains
  through World Models.* arXiv:2301.04104, 2023.

### 作者

陈杉杉（Shanshan Chen）
