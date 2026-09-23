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

项目需要**两个互不兼容的 Python 环境**，因为 Isaac Sim 只支持 Python 3.11，而 LeRobot 要求 Python 3.12。工具类 venv 是可选的辅助环境。**两个环境目前都已经搭好并且能正常用。**

| 角色 | 这台机器上用什么 | Python | 状态 |
|---|---|---|---|
| **isaaclab-env**（仿真、遥操作录制、RL 环境、评估） | conda 环境 **`isaac5.1-so-arm101`** | 3.11.15 | ✅ 本来就装好了 Isaac Sim 5.1.0.0 + Isaac Lab 2.3.0 + torch 2.7.0+cu128，直接复用 |
| **lerobot-env**（ACT/Diffusion/SmolVLA 训练、IQL critics、下载数据集） | conda 环境 **`lerobot-env`** | 3.12.14 | ✅ 新建的：LeRobot 0.6.2 + torch 2.11.0+cu128 + so101-lab |
| `venvs/rerun` | 数据集可视化 | 任意 | 可选，还没建 |
| `venvs/viewer` | 录制时摄像头实时预览 | 任意 | 可选，还没建 |

**注意：** 全程不需要 `uv`。两个环境都是用普通的 `conda create` + `pip` 搭的，`pyproject.toml` 里没有任何 uv 相关配置。

**注意：** `activate_isaaclab.sh` / `activate_lerobot.sh` 脚本假设的是有 `bin/activate` 的 venv，conda 环境没有这个文件，用不了。直接 `conda activate <环境名>` 就行。

### 怎么激活

```bash
# isaaclab-env — 仿真、遥操作、录制、评估
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab

# lerobot-env — BC/VLA 训练、下载/检查数据集
conda activate lerobot-env
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
```

---

## 3. 已经搭好的环境配置

### isaaclab-env（`isaac5.1-so-arm101`）

第一次跑 `record_episodes.py` 时缺这些包，已经装好了：
```bash
conda activate isaac5.1-so-arm101
pip install -e .
pip install pyarrow av datasets huggingface_hub accelerate num2words safetensors \
            cma nvidia-ml-py pyserial deepdiff feetech-servo-sdk diffusers
pip install "click==8.1.7"   # 锁定版本，原因见下面
```

**坑：click 版本锁定。** 装上面这批包时会把 `click` 自动升到比 `8.1.7` 新的版本，这会和 `isaacsim-kernel` 要求的精确版本（`click==8.1.7`）冲突，可能影响 Isaac Sim 本身，所以锁回去了。**副作用**：这个环境里 `typer`（进而 `trackio`）导入会失败——对应上游文档里"trackio 只能在 lerobot-env 用"这条已知限制。这个环境里 `--tracker` 永远用 `wandb` 或 `none`，别用 `trackio`。

### lerobot-env

Ubuntu 22.04 的 apt 源里没有 Python 3.12（得装 deadsnakes PPA 才有），所以这个环境改用 **conda** 建，而不是普通 venv——更省事，效果一样：
```bash
conda create -n lerobot-env python=3.12 -y
conda activate lerobot-env
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[dataset]"      # 必须带 `[dataset]` extra，光 `pip install -e .`
                                  # 不够——import lerobot.datasets.lerobot_dataset 会报
                                  # ImportError（缺 "datasets" 包）

cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
pip install -e ".[hardware]"
pip install trackio nvidia-ml-py
```

**坑：SOCKS 代理导致下不了 Hugging Face。** 这台机器全局开着 `ALL_PROXY=socks://127.0.0.1:7897`。`huggingface_hub` 用的 `httpx` 库只认 `socks5://`/`socks5h://` 这种带版本号的代理协议，不认裸的 `socks://`，所以下载会报 `ValueError: Unknown scheme for proxy URL`。同时设置的 `HTTPS_PROXY=http://127.0.0.1:7897` 本身是没问题的。解决办法是命令前去掉 `ALL_PROXY`：
```bash
env -u ALL_PROXY -u all_proxy python scripts/tools/download_demo.py --dataset easy --checkpoint easy
```
（一劳永逸的办法是把系统代理的 `ALL_PROXY` 改成 `socks5://127.0.0.1:7897`，这个还没改。）

### 硬件检查结果
GPU 是 RTX 4090，24GB 显存，远超项目要求的 8GB 最低门槛。`cmake` 是 apt 装的版本（必须是这个，snap 版会有问题）。

---

## 4. 官方示范数据集（已经下载好了，不用从零开始录数据也能上手）

上游项目在 Hugging Face 上放了预录制数据集 + 训练好的权重，这台机器上已经下载好了：

| 内容 | HF 仓库 | 本地路径 |
|---|---|---|
| Easy 数据集（200 条示范） | `MSSerg/so101-easy-task-v1` | `data/recordings/easy_task_v1` ✅ 已下载 |
| SmolVLA + IQL 权重（86-88% 成功率） | `MSSerg/so101-smolvla-iql-easy-v1` | `outputs/smolvla_iql_easy_v1` ✅ 已下载 |
| Medium 数据集（300 条） | `MSSerg/so101-figure-shape-placement-v1` | 没下，需要的话 `python scripts/tools/download_demo.py --dataset medium` |

可以跳过自己录数据，直接用：
```bash
conda activate lerobot-env
python scripts/train/train_act.py --dataset data/recordings/easy_task_v1 --name my_act_v1

# 或者直接评估下载好的权重（两个终端）
# 终端1（lerobot-env）：
python scripts/eval/smolvla_server.py --checkpoint outputs/smolvla_iql_easy_v1
# 终端2（isaaclab-env）：
conda activate isaac5.1-so-arm101
python scripts/eval/eval_vla_policy.py --checkpoint outputs/smolvla_iql_easy_v1 \
    --env figure_shape_placement_easy --episodes 10 --no-domain-rand --gui
```

---

## 5. 日常使用流程

### 录制数据集（isaaclab-env）
```bash
conda activate isaac5.1-so-arm101

# 键盘控制（不需要实体机械臂）——不要加 --headless，键盘输入依赖 Isaac Sim 窗口
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=keyboard

# 用实体 SO-101 leader 臂——完整硬件操作流程见下面第 7 节
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader --calibration-file=leader_1.json --port=/dev/ttyACM0 \
    --reward-mode success
```

### 训练模仿学习策略（lerobot-env）
```bash
conda activate lerobot-env
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

## 6. USD → STL 转换（打印真实道具用）

任务道具（`assets/props/*.usd` —— 立方体、五角星、六边形、圆柱等）只有 USD 格式，没有 STL 原始文件（机械臂本体反而自带 STL，在 `assets/robots/so101/meshes/*.stl`）。写了个 `scripts/tools/usd_to_stl.py` 来转换——把每个网格的世界变换烘焙进去，并把场景单位（米）转成毫米，转出来的 STL 就是实际打印尺寸。

```bash
conda activate isaac5.1-so-arm101
python scripts/tools/usd_to_stl.py assets/props --output assets/props_stl
```

转换结果（都是仿真里实际用的尺寸，不用再缩放）：

| 道具 | 尺寸（mm） | 能不能打印 |
|---|---|---|
| light_blue_cube | 40.0 × 40.0 × 16.5 | ✅ |
| deep_green_star | 42.2 × 44.0 × 16.5 | ✅ |
| deep_blue_hexagon | 39.8 × 45.4 × 16.5 | ✅ |
| yellow_cylinder | 42.0 × 42.0 × 16.5 | ✅ |
| light_green_triangle_prism | 40.8 × 45.3 × 16.5 | ✅ |
| orange_x | 40.5 × 40.5 × 16.5 | ✅ |
| pink_diamond | 53.5 × 40.2 × 16.5 | ✅ |
| red_clover | 41.0 × 41.0 × 16.5 | ✅ |
| yellow_parallelepiped | 45.0 × 35.0 × 16.5 | ✅ |
| platfrom_full（目标托盘） | 200 × 200 × 9 | ✅（桌面打印机尺寸内） |
| full_game（合并场景） | 50000 × 50000 × 23.4 | ⚠️ 含一个巨大背景平面，不是实际道具尺寸，别打印这个 |
| workbench（工作台） | 900 × 600 × 750 | ⚠️ 真实家具尺度，仅供参考 |

---

## 7. 真实 leader 臂硬件遥操作

已经在这台机器上完整验证跑通：`so101leader` 设备直接读取真实 leader 臂的关节角度（扭矩关闭、可以手动自由掰动），1:1 映射到仿真机械臂的关节——没有 IK，纯关节镜像。只需要接 leader 臂（仿真本身就是 follower）。

### 一次性：修串口权限
`/dev/ttyACM0` 属主是 `root:dialout`（权限 660）。账号本身在 `dialout` 组里，但是**组权限只对"加入该组之后新开的会话"生效**——已经开着的终端不会自动获得。解决办法（二选一）：
```bash
newgrp dialout          # 当前终端里开一个子 shell，立刻拿到 dialout 权限
# 或者直接开一个新终端窗口
```

### 一次性：校准
```bash
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
python scripts/calibrate_leader.py --port /dev/ttyACM0 --name leader_1
```
把臂摆到中间位置回车 → 把每个关节都转到极限位置回车 → 存到 `so101_lab/devices/lerobot/.cache/leader_1.json`。

### 测试控制（不录制）
```bash
python scripts/teleop/teleop_agent.py \
    --task figure_shape_placement_easy \
    --teleop-device so101leader \
    --port /dev/ttyACM0 \
    --calibration-file leader_1.json \
    --episode-length 120
```
**坑 1：** `teleop_agent.py` 的任务参数是 `--task`，不是 `--env`——跟 `record_episodes.py`（用的是 `--env`）不一致，两个脚本之间很容易记混。

**坑 2：** `teleop_agent.py` 的 `--episode-length` **默认只有 25 秒**——这是给快速测试联动效果用的默认值，对还在熟悉操作手感的阶段明显不够完成一次完整的抓取摆放。环境重置是单纯计时到点（`env.episode_length_buf[0] >= env.max_episode_length`），跟有没有检测到"放置失败"没关系。练手的时候把这个参数调大（`--episode-length 120` 或更长）。

### 正式录制
```bash
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port=/dev/ttyACM0 \
    --reward-mode success
```
（`record_episodes.py` 本身默认就是 120 秒一条，不像 `teleop_agent.py` 那么短。）

---

## 8. 键盘遥操作控制表

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

## 9. 这台机器上已知会遇到的问题

| 现象 | 原因 | 解决方法 |
|---|---|---|
| isaaclab-env 里 `ModuleNotFoundError: No module named 'pyarrow'`（或 `av`、`datasets`） | isaaclab-env 的"首次额外依赖"这步没做 | 已经装好了，见第 3 节 |
| `[WARNING] Viewer venv not found` | 可选的 `venvs/viewer` 预览环境没建 | 不影响功能，忽略即可；想要预览就 `python3 -m venv venvs/viewer && source venvs/viewer/bin/activate && pip install opencv-python numpy` |
| isaaclab-env 里 `trackio`/`typer` 导入报错 | `click` 被锁定在 8.1.7 以保证和 Isaac Sim 兼容 | 属于预期行为——isaaclab-env 里用 `--tracker wandb` 或 `none`，`--tracker trackio` 只能在 lerobot-env 用 |
| 键盘按键没反应 | 用了 `--headless`（没窗口收不到按键）；或者忘了先按 `Space` | 键盘遥操作别加 `--headless`；先按 `Space` 再按其他键 |
| `ModuleNotFoundError: No module named 'lerobot'` | 在 isaaclab-env 里跑了只能在 lerobot-env 跑的脚本（比如 `download_demo.py`、`train_act.py`） | 换成 `conda activate lerobot-env` |
| `import lerobot.datasets...` 报 `ImportError: 'datasets' is required` | 装 lerobot 时用了普通 `pip install -e .`，没带 `[dataset]` extra | 在 lerobot 仓库根目录 `pip install -e ".[dataset]"` |
| 下载 HF 数据集报 `ValueError: Unknown scheme for proxy URL 'socks://...'` | 系统 `ALL_PROXY` 用的是裸 `socks://`，httpx 只认 `socks5://`/`socks5h://` | 命令前加 `env -u ALL_PROXY -u all_proxy` |
| `PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'` | 当前终端会话是在加入 `dialout` 组之前开的 | `newgrp dialout` 或者开个新终端 |
| `teleop_agent.py: error: unrecognized arguments: --env` | 这个脚本的参数是 `--task`，不是 `--env`（和 `record_episodes.py` 不一样） | 改用 `--task <环境名>` |
| `teleop_agent.py` 里抓取摆放做到一半仿真就重置了 | `--episode-length` 默认 25 秒（快速测试用的默认值），单纯计时到点，不是检测失败 | 加 `--episode-length 120` 或更长 |

这里没覆盖到的问题去查 [docs/troubleshooting.md](docs/troubleshooting.md)。

---

## 10. 想深入了解去哪看

| 需求 | 文件 |
|---|---|
| 按任务查怎么做（"我想录制/训练/评估/部署"） | [docs/guide.md](docs/guide.md) |
| 完整分步工作流 | [docs/workflow/](docs/workflow/) |
| 为什么要 4 个环境、数据格式、时序设计 | [docs/architecture.md](docs/architecture.md) |
| 每个脚本的参数详解 | [docs/reference/index.md](docs/reference/index.md) |
| 实验结果详情 | [docs/results/](docs/results/) |
