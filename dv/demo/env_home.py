"""
居家场景仿真环境 - HomeEnv

模拟多层住宅中的服务机器人导航任务：
- 从房间出发，经过走廊到达电梯区域
- 支持 RGB 图像观测和离散动作
- 兼容 OpenAI Gym 接口，可直接接入 DreamerV3 的 FromGym 包装器

任务链：Room1 → Corridor → Elevator → Corridor → Room2 (Goal)
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from PIL import Image, ImageDraw


# === 居家环境布局定义 ===
# 符号: R=房间1(起点), G=房间2(终点), E=电梯区, .=走廊, #=墙
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

HOME_LAYOUT_MEDIUM = [
    "RRRR..............",
    "RRRR..............",
    "RRRR..............",
    "RRRR..............",
    "...........EEEE...",
    "...........EEEE...",
    "...........EEEE...",
    "...........EEEE...",
    "...........EEEE...",
    "######.............",
    "........######....",
    "................GG",
    "................GG",
    "................GG",
    "................GG",
]

# 任务阶段定义
PHASE_TO_ROOM = "to_elevator"       # 从房间去电梯
PHASE_IN_ELEVATOR = "in_elevator"   # 在电梯轿厢内
PHASE_TO_GOAL = "to_goal"           # 从电梯去目标房间
PHASE_DONE = "done"


class HomeEnv(gym.Env):
    """居家服务机器人导航仿真环境"""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 10}

    def __init__(self, layout="small", size=64, max_steps=200,
                 render_mode="rgb_array", hard_mode=False):
        super().__init__()
        self.layout_name = layout
        self.size = size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.hard_mode = hard_mode

        # 加载布局
        layout_map = HOME_LAYOUT_SMALL if layout == "small" else HOME_LAYOUT_MEDIUM
        self.grid = self._parse_layout(layout_map)
        self.grid_h, self.grid_w = self.grid.shape

        # 查找特殊位置
        self.room1_positions = self._find_positions("R")
        self.room2_positions = self._find_positions("G")
        self.elevator_positions = self._find_positions("E")
        self.wall_positions = self._find_positions("#")

        # 动作空间: 0=上, 1=下, 2=左, 3=右, 4=停留
        self.action_space = spaces.Discrete(5)

        # 观测空间: RGB 图像
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(size, size, 3), dtype=np.uint8
        )

        # 电梯内部区域（电梯框内的位置）
        self._elevator_interior = self._compute_elevator_interior()

        # 当前状态
        self.agent_pos = None
        self.phase = None
        self.step_count = 0
        self.visited = set()
        self.reached_elevator = False
        self.reached_goal = False
        self.trajectory = []

    def _parse_layout(self, layout_lines):
        """将字符串布局转为numpy数组，自动对齐行宽度"""
        h = len(layout_lines)
        w = max(len(line) for line in layout_lines)  # 取最大行宽
        grid = np.zeros((h, w), dtype=np.uint8)
        char_to_id = {".": 0, "R": 1, "G": 2, "E": 3, "#": 4}
        for i, line in enumerate(layout_lines):
            for j, ch in enumerate(line):
                grid[i, j] = char_to_id.get(ch, 0)
        return grid

    def _find_positions(self, char):
        """在原始布局中查找指定字符的位置"""
        layout_map = HOME_LAYOUT_SMALL if self.layout_name == "small" else HOME_LAYOUT_MEDIUM
        positions = []
        for i, line in enumerate(layout_map):
            for j, ch in enumerate(line):
                if ch == char:
                    positions.append((i, j))
        return positions

    def _compute_elevator_interior(self):
        """计算电梯轿厢内部位置"""
        if not self.elevator_positions:
            return set()
        # 找电梯区域的边界框内部
        rows = [p[0] for p in self.elevator_positions]
        cols = [p[1] for p in self.elevator_positions]
        interior = set()
        for r in range(min(rows), max(rows) + 1):
            for c in range(min(cols), max(cols) + 1):
                if self.grid[r, c] == 3:  # E
                    interior.add((r, c))
        return interior

    def _random_start_pos(self):
        """在房间1中随机选择起始位置"""
        return self.room1_positions[np.random.randint(len(self.room1_positions))]

    def _is_wall(self, r, c):
        if 0 <= r < self.grid_h and 0 <= c < self.grid_w:
            return self.grid[r, c] == 4  # #
        return True  # 边界外视为墙

    def _move(self, pos, action):
        """执行移动，返回新位置"""
        r, c = pos
        moves = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1), 4: (0, 0)}
        dr, dc = moves[action]
        nr, nc = r + dr, c + dc
        if self._is_wall(nr, nc):
            return pos  # 撞墙，不移动
        return (nr, nc)

    def _check_phase_transition(self, pos):
        """检查是否需要切换任务阶段"""
        new_phase = self.phase

        if self.phase == PHASE_TO_ROOM:
            if pos in self._elevator_interior:
                new_phase = PHASE_IN_ELEVATOR
                self.reached_elevator = True

        elif self.phase == PHASE_IN_ELEVATOR:
            if pos not in self._elevator_interior:
                new_phase = PHASE_TO_GOAL

        elif self.phase == PHASE_TO_GOAL:
            if pos in set(self.room2_positions):
                new_phase = PHASE_DONE
                self.reached_goal = True

        return new_phase

    def _compute_reward(self, pos, prev_phase, new_phase, prev_pos):
        """计算奖励"""
        reward = 0.0

        # 到达电梯内部
        if prev_phase == PHASE_TO_ROOM and new_phase == PHASE_IN_ELEVATOR:
            reward += 5.0

        # 进入目标房间
        if prev_phase == PHASE_TO_GOAL and new_phase == PHASE_DONE:
            reward += 10.0

        # 探索奖励：访问新区域
        if pos not in self.visited:
            reward += 0.1

        # 移动惩罚（鼓励高效路径）
        if pos == prev_pos:
            reward -= 0.05  # 撞墙/停留更重的惩罚
        else:
            reward -= 0.02

        # 靠近目标的额外奖励（启发式引导）
        if self.phase == PHASE_TO_ROOM and self.elevator_positions:
            target = np.mean(self.elevator_positions, axis=0)
            prev_dist = np.linalg.norm(np.array(prev_pos) - target)
            curr_dist = np.linalg.norm(np.array(pos) - target)
            reward += (prev_dist - curr_dist) * 0.05

        elif self.phase == PHASE_TO_GOAL and self.room2_positions:
            target = np.mean(self.room2_positions, axis=0)
            prev_dist = np.linalg.norm(np.array(prev_pos) - target)
            curr_dist = np.linalg.norm(np.array(pos) - target)
            reward += (prev_dist - curr_dist) * 0.05

        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = self._random_start_pos()
        self.phase = PHASE_TO_ROOM
        self.step_count = 0
        self.visited = {self.agent_pos}
        self.reached_elevator = False
        self.reached_goal = False
        self.trajectory = [self.agent_pos]
        return self._get_obs(), {}

    def step(self, action):
        action = int(action)  # Gymnasium wrapper may pass numpy scalar
        prev_pos = self.agent_pos
        prev_phase = self.phase

        # 执行移动
        self.agent_pos = self._move(self.agent_pos, action)
        self.step_count += 1
        self.visited.add(self.agent_pos)
        self.trajectory.append(self.agent_pos)

        # 检查阶段转移
        self.phase = self._check_phase_transition(self.agent_pos)

        # 计算奖励
        reward = self._compute_reward(
            self.agent_pos, prev_phase, self.phase, prev_pos
        )

        # 终止条件
        terminated = self.phase == PHASE_DONE
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {
            "phase": self.phase,
            "pos": self.agent_pos,
            "reached_elevator": self.reached_elevator,
            "reached_goal": self.reached_goal,
            "steps": self.step_count,
            "coverage": len(self.visited),
            "trajectory": self.trajectory.copy(),
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """渲染当前帧为RGB图像"""
        return self.render()

    def render(self):
        """渲染环境为64x64 RGB图像"""
        img = Image.new("RGB", (self.size, self.size), color=(240, 240, 245))
        draw = ImageDraw.Draw(img)

        cell_w = self.size / self.grid_w
        cell_h = self.size / self.grid_h

        # 绘制网格
        colors = {
            0: (245, 245, 250),   # 走廊 - 浅灰
            1: (200, 220, 255),   # 房间1 - 浅蓝
            2: (200, 255, 220),   # 房间2 - 浅绿（目标）
            3: (255, 230, 200),   # 电梯 - 浅橙
            4: (120, 100, 100),   # 墙 - 深灰
        }

        for i in range(self.grid_h):
            for j in range(self.grid_w):
                cell_id = self.grid[i, j]
                color = colors.get(cell_id, (240, 240, 240))
                x0 = int(j * cell_w)
                y0 = int(i * cell_h)
                x1 = int((j + 1) * cell_w)
                y1 = int((i + 1) * cell_h)
                draw.rectangle([x0, y0, x1, y1], fill=color, outline=(220, 220, 225))

        # 绘制电梯门（横线标记）
        if self.elevator_positions:
            e_top = min(p[0] for p in self.elevator_positions)
            e_left = min(p[1] for p in self.elevator_positions)
            e_right = max(p[1] for p in self.elevator_positions)
            draw.line(
                [(int(e_left * cell_w), int(e_top * cell_h)),
                 (int((e_right + 1) * cell_w), int(e_top * cell_h))],
                fill=(200, 100, 100), width=2
            )

        # 绘制历史轨迹
        for idx, (tr, tc) in enumerate(self.trajectory):
            if idx == 0:
                continue
            prev_r, prev_c = self.trajectory[idx - 1]
            alpha = 0.3 + 0.7 * (idx / max(len(self.trajectory), 1))
            r_val = int(100 * (1 - alpha) + 50 * alpha)
            g_val = int(100 * (1 - alpha) + 150 * alpha)
            b_val = int(200 * (1 - alpha) + 255 * alpha)
            draw.line(
                [(int((prev_c + 0.5) * cell_w), int((prev_r + 0.5) * cell_h)),
                 (int((tc + 0.5) * cell_w), int((tr + 0.5) * cell_h))],
                fill=(r_val, g_val, b_val), width=2
            )

        # 绘制智能体
        ar, ac = self.agent_pos
        cx = int((ac + 0.5) * cell_w)
        cy = int((ar + 0.5) * cell_h)
        radius = max(3, int(min(cell_w, cell_h) * 0.35))
        # 外圈
        draw.ellipse(
            [cx - radius - 1, cy - radius - 1, cx + radius + 1, cy + radius + 1],
            fill=(50, 50, 50)
        )
        # 内圈
        agent_color = (255, 80, 80)  # 红色
        if self.phase == PHASE_IN_ELEVATOR:
            agent_color = (100, 100, 255)  # 电梯内蓝色
        elif self.phase == PHASE_DONE:
            agent_color = (80, 255, 80)  # 完成绿色
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=agent_color
        )

        # 方向指示
        draw.line(
            [(cx, cy), (cx, cy - radius)],
            fill=(255, 255, 255), width=1
        )

        return np.array(img, dtype=np.uint8)

    def close(self):
        pass


# 注册 Gym 环境
gym.register(
    id="HomeNav-v0",
    entry_point="demo.env_home:HomeEnv",
    kwargs={"layout": "small", "size": 64, "max_steps": 200},
)

gym.register(
    id="HomeNav-Medium-v0",
    entry_point="demo.env_home:HomeEnv",
    kwargs={"layout": "medium", "size": 64, "max_steps": 300},
)

if __name__ == "__main__":
    # 测试渲染
    env = HomeEnv(layout="small", size=256, max_steps=200)
    obs, _ = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")

    # 运行几步并保存截图
    frames = [obs]
    for _ in range(30):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(obs)
        if terminated or truncated:
            break

    # 保存GIF
    gif_frames = [Image.fromarray(f) for f in frames]
    output_path = "home_env_demo.gif"
    gif_frames[0].save(
        output_path, save_all=True, append_images=gif_frames[1:],
        duration=200, loop=0
    )
    print(f"Saved demo GIF to {output_path}")
    print(f"Final info: {info}")
