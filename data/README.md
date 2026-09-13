# 数据集说明 / Dataset

本项目采用强化学习在线生成范式：数据并非预先采集标注的静态数据集，而是在
智能体与环境实时交互过程中动态产生的训练日志与模型状态。

## 数据文件

| 文件 | 说明 | 大小 |
|---|---|---|
| `metrics.jsonl` | 完整训练指标日志（约 18,500 条 Episode 记录 + 训练报告数据点） | ~1.8 MB |
| `scores.jsonl` | Episode Score 精简日志 | ~1.0 MB |
| `sample_trajectories.json` | 从训练中提取的示例轨迹（含随机策略与训练后策略） | ~4 KB |

## metrics.jsonl 字段说明

每行一个 JSON 对象，主要包含以下键：

- `episode/score`：单轮 Episode 的累计奖励（Score）
- `episode/length`：单轮 Episode 的决策步数
- `train/loss/dyn`：动力学 KL 散度损失
- `train/loss/rep`：表征 KL 散度损失
- `train/loss/reward`：奖励预测损失
- `train/loss/image`：图像重建损失
- `train/rand/action`：随机动作比例（探索率）
- `fps/train`、`fps/policy`：训练与推理吞吐量

## 模型权重

训练好的完整模型权重（`agent.pkl`，约 123 MB）因体积较大，未包含在本仓库中，
而是以 GitHub Release 资产的形式发布。请前往
[Releases 页面](https://github.com/REPO_OWNER/REPO_NAME/releases) 下载，
或通过重新运行训练复现（约 6–8 小时，RTX 4060 8GB）。

## 复现方式

```bash
cd code
set XLA_PYTHON_CLIENT_PREALLOCATE=false
python main.py --configs home4060 --logdir ./logs
```
