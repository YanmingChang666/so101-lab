# SO-101 Lab — 个人学习笔记

> 这是针对**你这台机器**的个性化速查文档，目的是"照着抄命令就能跑起来"。完整的上游项目文档（架构设计、每个脚本的参数、实验结果详表、引用）在 [docs/](docs/) 里，需要深入细节时去那边查。

---

## 1. 这个项目是干什么的

让 SO-101 机械臂在仿真里通过"看人示范"学会抓取/摆放物体，再用强化学习进一步提升成功率，最终可以部署到真实机械臂上：

```
Isaac Sim 里遥操作  →  录制 LeRobot 格式数据集  →  训练模仿学习策略 (ACT / Diffusion / SmolVLA)
                                                  →  仿真里评估
                                                  →  （可选）RL 微调 (IQL 离线 / PPO 在线)
                                                  →  部署到真实 SO-101
```

任务场景：工作台上放着一套儿童形状分类玩具（立方体、五角星、六边形、圆柱等），机械臂要把物体夹起来放到目标位置。有两档难度，对应两个不同的任务环境：
- `figure_shape_placement_easy` — 出生点范围小，**建议从这个开始**
- `figure_shape_placement` — 出生点范围是 easy 的 4 倍，更难

上游项目参考成功率（SmolVLA + IQL 微调）：easy 任务 86–88%，medium 任务 48%。纯模仿学习（ACT）大概只有 56%。

---

## 2. 这台机器上用哪些环境

项目需要**两个互不兼容的 Python 环境**，因为 Isaac Sim 只支持 Python 3.11，而 LeRobot 0.5.1 要求 Python 3.12。工具类 venv 是可选的辅助环境。

| 角色 | 这台机器上用什么 | Python | 状态 |
|---|---|---|---|
| **isaaclab-env**（仿真、遥操作录制、RL 环境、评估） | conda 环境 **`isaac5.1-so-arm101`** | 3.11.15 | ✅ 已经装好 Isaac Sim 5.1.0.0 + Isaac Lab 2.3.0 + torch 2.7.0+cu128，直接复用，不用新建 |
| **lerobot-env**（ACT/Diffusion/SmolVLA 训练、IQL critics） | 普通 `python3.12 -m venv`（不用 conda，也不需要 uv） | 3.12 | 还没建，见下面 |
| `venvs/rerun` | 数据集可视化 | 任意 | 可选 |
| `venvs/viewer` | 录制时摄像头实时预览 | 任意 | 可选 |

**注意：** 全程不需要 `uv`，官方文档用它只是图快。普通 `pip`/`venv`/`conda` 效果完全一样，`pyproject.toml` 里没有任何 uv 相关配置。

**注意：** `activate_isaaclab.sh` 脚本假设的是有 `bin/activate` 的 venv，conda 环境没有这个文件，用不了。直接 `conda activate isaac5.1-so-arm101` 就行。

### 怎么激活

```bash
# isaaclab-env
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab

# lerobot-env（建好之后，见下面搭建步骤）
source ~/venvs/lerobot-env/bin/activate
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
```

---

## 3. `isaac5.1-so-arm101` 环境里已经做过的配置

第一次跑 `record_episodes.py` 时缺这些包，已经装好了：

```bash
conda activate isaac5.1-so-arm101
pip install -e .
pip install pyarrow av datasets huggingface_hub accelerate num2words safetensors \
            cma nvidia-ml-py pyserial deepdiff feetech-servo-sdk diffusers
pip install "click==8.1.7"   # 锁定版本，原因见下面
```

### 坑：click 版本锁定
装上面这批包时会把 `click` 自动升到比 `8.1.7` 新的版本，这会和 `isaacsim-kernel` 要求的精确版本（`click==8.1.7`）冲突，可能影响 Isaac Sim 本身，所以锁回去了。**副作用**：这个环境里 `typer`（进而 `trackio`）导入会失败——这正好对应上游文档里说的"trackio 只能在 lerobot-env 用"这条已知限制。在 isaaclab-env 里跑脚本时，`--tracker` 永远用 `wandb` 或 `none`，别用 `trackio`。

### lerobot-env — 还没建，需要你自己搭
```bash
python3.12 -m venv ~/venvs/lerobot-env
source ~/venvs/lerobot-env/bin/activate
pip install --upgrade pip
git clone https://github.com/huggingface/lerobot.git ~/lerobot   # 需要 0.5.1 以上版本
cd ~/lerobot && pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
pip install -e ".[hardware]"
pip install trackio nvidia-ml-py
```

然后配置 `.env`（从 `.env.example` 复制）：
```
LEROBOT_ENV=/home/cym/venvs/lerobot-env
LEROBOT_SRC=/home/cym/lerobot/src
```
（`ISAACLAB_ENV` 这项用不上，因为用的是 conda，直接手动 `conda activate isaac5.1-so-arm101` 代替 `act-isaac` 别名。）

### 硬件检查结果
GPU 是 RTX 4090，24GB 显存，远超项目要求的 8GB 最低门槛。`cmake` 是 apt 装的版本（必须是这个，snap 版会有问题）。

---

## 4. 日常使用流程

### 录制数据集（isaaclab-env）
```bash
conda activate isaac5.1-so-arm101

# 键盘控制（不需要实体机械臂）——不要加 --headless，键盘输入依赖 Isaac Sim 窗口
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=keyboard

# 用实体 SO-101 leader 臂
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader --calibration-file=leader_1.json --port=/dev/ttyACM0 \
    --reward-mode success
```

不想自己录，直接下载官方示范数据集+权重：
```bash
python scripts/tools/download_demo.py --dataset easy --checkpoint easy
```

### 训练模仿学习策略（lerobot-env）
```bash
source ~/venvs/lerobot-env/bin/activate
python scripts/train/train_act.py \
    --dataset data/recordings/my_task_v1 \
    --config configs/policy/act/baseline.yaml \
    --name act_v1 --tracker wandb
```

### 仿真里评估（isaaclab-env）
```bash
conda activate isaac5.1-so-arm101
python scripts/eval/eval_act_policy.py \
    --checkpoint outputs/act_v1 \
    --env figure_shape_placement_easy \
    --episodes 20 --max-steps 500 --headless
```

### RL 微调（可选，成功率能再上一个台阶）
参考 [docs/workflow/04_offline_rl.md](docs/workflow/04_offline_rl.md)（IQL，建议先做这个）和 [docs/workflow/05_online_rl.md](docs/workflow/05_online_rl.md)（PPO）。

---

## 5. 键盘遥操作控制表

录制脚本启动时会打印这张表。大部分按键控制的是**夹爪在自身坐标系下的笛卡尔增量位移**（不是直接控制关节角度），只有 `A`/`D` 和 `U`/`O` 是直接控制关节。

| 按键 | 作用 |
|---|---|
| `Space` | 启动遥操作（**必须先按这个**，不按的话其他键都没反应） |
| `N` | 开始录制这一条 episode |
| `F` | 结束并保存 |
| `R` | 重启（重置到同一场景） |
| `X` | 丢弃这条，重置 |
| `T` | 调试信息 |
| `Escape` | 退出 |
| `W` / `S` | 夹爪前进 / 后退 |
| `A` / `D` | 底座左转 / 右转（shoulder_pan 关节） |
| `Q` / `E` | 夹爪上移 / 下移 |
| `I` / `K` | 俯仰上 / 下 |
| `J` / `L` | 偏航左 / 右 |
| `U` / `O` | 夹爪张开 / 闭合 |

按键是**按住持续移动、松开停止**（不是点一下动一下）。速度可以用 `--sensitivity <数字>` 调（默认 1.0）。窗口焦点必须在 Isaac Sim 仿真窗口上，不能在终端里按。

典型操作顺序：`Space` → 先移动几下热身 → 需要的话 `X`/`R` 重置 → `N` → 完成一次抓取摆放 → 成功按 `F` 保存（失败按 `X` 丢弃）→ 重复 50~200 次 → `Escape` 退出。

---

## 6. 这台机器上已知会遇到的问题

| 现象 | 原因 | 解决方法 |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyarrow'`（或 `av`、`datasets`） | isaaclab-env 的"首次额外依赖"这步没做（setup 文档第 3 节） | 已经装好了，见上面第 3 节 |
| `[WARNING] Viewer venv not found` | 可选的 `venvs/viewer` 预览环境没建 | 不影响功能，忽略即可；想要预览就 `python3 -m venv venvs/viewer && source venvs/viewer/bin/activate && pip install opencv-python numpy` |
| isaaclab-env 里 `trackio`/`typer` 导入报错 | `click` 被锁定在 8.1.7 以保证和 Isaac Sim 兼容 | 属于预期行为——isaaclab-env 里用 `--tracker wandb` 或 `none`，`--tracker trackio` 只能在 lerobot-env 用 |
| 键盘按键没反应 | 用了 `--headless`（没窗口收不到按键）；或者忘了先按 `Space` | 键盘遥操作别加 `--headless`；先按 `Space` 再按其他键 |

这里没覆盖到的问题去查 [docs/troubleshooting.md](docs/troubleshooting.md)。

---

## 7. 想深入了解去哪看

| 需求 | 文件 |
|---|---|
| 按任务查怎么做（"我想录制/训练/评估/部署"） | [docs/guide.md](docs/guide.md) |
| 完整分步工作流 | [docs/workflow/](docs/workflow/) |
| 为什么要 4 个环境、数据格式、时序设计 | [docs/architecture.md](docs/architecture.md) |
| 每个脚本的参数详解 | [docs/reference/index.md](docs/reference/index.md) |
| 实验结果详情 | [docs/results/](docs/results/) |
