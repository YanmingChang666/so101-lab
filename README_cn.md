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

任务道具（`assets/props/*.usd` —— 立方体、五角星、六边形、圆柱等）只有 USD 格式，没有 STL 原始文件（机械臂本体反而自带 STL，在 `assets/robots/so101/meshes/*.stl`）。写了个 `scripts/tools/usd_to_stl.py` 来转换——把每个网格的世界变换烘焙进去，并按 USD 场景单位换算为米（默认 `--units m`）。STL 本身不记录单位，导入软件时应选择米、缩放设为 1；用于 3D 打印时加 `--units mm`，建议输出到单独的 `assets/props_stl_mm` 目录。

```bash
conda activate isaac5.1-so-arm101
python scripts/tools/usd_to_stl.py assets/props --output assets/props_stl
```

物理尺寸如下（表中以毫米展示；默认导出的 STL 坐标以米表示，例如立方体为 `0.04 × 0.04 × 0.0165`）：

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

### 连接后：确认实际串口

先激活环境，在项目目录中查看设备：

```bash
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

有多个设备时，拔下再插上 leader 的 USB，对比新增的端口。`ttyACM0` 的编号可能变化，也可能显示为 `ttyUSB0`；以实际输出为准。旧报错提示中的 `python lerobot/find_port.py` 在此项目中不存在，使用上面的命令即可。

本机曾识别到序列号 `5A7A057348`，固定设备链接如下。建议用它代替易变化的端口编号；其他机械臂需替换为自己的链接：

```bash
export LEADER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A057348-if00
ls -l "$LEADER_PORT"
```

后续命令在同一个终端执行；新开终端需重新设置 `LEADER_PORT`。固定链接仍依赖 USB 正常连接，无法修复掉线。

### 按报错排查连接与权限

| 报错 | 含义与处理 |
|---|---|
| `No such file or directory` / `Errno 2` | 指定设备不存在。重新查看串口，检查 USB 数据线、供电和连接，重新插拔；`chmod` 不能解决设备不存在。 |
| `Permission denied` / `Errno 13` | 设备存在，但当前进程缺少权限。按下面步骤检查 `dialout`。 |
| `python: can't open file ... lerobot/find_port.py` | 工具路径不存在，改用 `python -m serial.tools.list_ports -v`。 |

`ls -l "$LEADER_PORT"` 查看链接，`ls -lL "$LEADER_PORT"` 查看实际设备权限。串口通常属于 `root:dialout`，权限为 660。

```bash
id -nG          # 当前 shell 的有效组
id -nG "$USER"  # 账号配置中的组
# 仅账号尚未加入 dialout 时执行：
sudo usermod -aG dialout "$USER"
# 账号已加入、当前 shell 尚未生效时：
newgrp dialout
conda activate isaac5.1-so-arm101
```

也可以注销桌面账号后重新登录，让整个会话获得新权限。**仅打开新终端不一定生效**，因为它可能继承旧桌面会话的组。无需使用 `chmod 666` 向所有用户开放串口。

### 一次性：校准

同一条 leader 已有有效的 `leader_1.json` 时，重新插拔 USB 或端口编号改变通常不需要重新校准，可以直接进入测试控制。仅首次使用或需要重新标定时执行下面命令；校准时缓慢移动到正常机械行程边界，不要强行掰动。

```bash
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
python scripts/calibrate_leader.py --port "$LEADER_PORT" --name leader_1
```
把臂摆到中间位置回车 → 把每个关节都转到极限位置回车 → 存到 `so101_lab/devices/lerobot/.cache/leader_1.json`。

### 测试控制（不录制）
```bash
python scripts/teleop/teleop_agent.py \
    --task figure_shape_placement_easy \
    --teleop-device so101leader \
    --port "$LEADER_PORT" \
    --calibration-file leader_1.json \
    --episode-length 300
```
窗口打开后聚焦 Isaac Sim 窗口，按 `Space` 启动遥操作，缓慢移动 leader 检查联动。

**坑 1：** `teleop_agent.py` 的任务参数是 `--task`，不是 `--env`——跟 `record_episodes.py`（用的是 `--env`）不一致，两个脚本之间很容易记混。

**坑 2：** `teleop_agent.py` 的 `--episode-length` **默认只有 25 秒**——这是给快速测试联动效果用的默认值，对还在熟悉操作手感的阶段明显不够完成一次完整的抓取摆放。环境重置是单纯计时到点（`env.episode_length_buf[0] >= env.max_episode_length`），跟有没有检测到"放置失败"没关系。练手的时候把这个参数调大（`--episode-length 120` 或更长）。

### 正式录制
```bash
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port="$LEADER_PORT" \
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
| `PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'` | 当前终端会话是在加入 `dialout` 组之前开的 | `newgrp dialout` 或注销桌面后重新登录（仅新开终端不一定生效） |
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

---

## 11. 下载数据集、训练 ACT 与可视化

以下命令均在项目根目录执行：
```bash
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
```

### 11.1 数据集位置与下载

| 数据集 | Hugging Face 仓库 | 本地目录 | 对应评估环境 |
|---|---|---|---|
| Easy | `MSSerg/so101-easy-task-v1` | `data/recordings/easy_task_v1` | `figure_shape_placement_easy` |
| Medium | `MSSerg/so101-figure-shape-placement-v1` | `data/recordings/figure_shape_placement_v1` | `figure_shape_placement` |

本地 Easy 元数据记录了 200 条示范、74,826 帧、30 FPS。完整路径为 `/home/cym/ROS/Arms/isaac_so_arm101/so101-lab/data/recordings/easy_task_v1`。

```bash
conda activate lerobot-env
# 下载 Easy；换成 medium 下载更难任务的数据，all 下载两套数据
# 去掉 ALL_PROXY，避开本机 socks:// 代理协议兼容问题
env -u ALL_PROXY -u all_proxy python scripts/tools/download_demo.py --dataset easy

# 检查目录、元数据、LeRobot 加载及首个样本读取
python scripts/verify_lerobot_dataset.py data/recordings/easy_task_v1
```

下载脚本发现目标目录非空就会跳过，**不会检查完整性或自动修复未完成下载**。已有数据可以先运行检查脚本。

```text
 easy_task_v1/
 ├── meta/       数据集信息、统计量、任务和 episode 索引
 ├── data/       Parquet：关节状态、动作、奖励等
 └── videos/
     ├── observation.images.top/chunk-000/file-000.mp4
     └── observation.images.wrist/chunk-000/file-000.mp4
```

每个 MP4 可以包含多条示范，不能直接把文件编号当成 episode 编号。

### 11.2 正式训练与短流程验证

```bash
conda activate lerobot-env
env -u ALL_PROXY -u all_proxy \
  python scripts/train/train_act.py \
    --dataset data/recordings/easy_task_v1 \
    --config configs/policy/act/baseline.yaml \
    --name act_easy_v1 \
    --num-workers 4 \
    --best-check-interval 100 \
    --tracker none
```

命令行参数覆盖 YAML，YAML 覆盖脚本默认值。上面的实际配置是：

| 参数 | 值 | 用途 |
|---|---|---|
| `steps` | 60000 | 优化器更新总次数，不是 epoch 数 |
| `batch_size` | 24 | 每批样本数；显存不足可加 `--batch-size 8` |
| `lr` | `1e-5` | 学习率 |
| `chunk_size` / `n_action_steps` | 25 / 25 | 预测的动作序列长度 / 每次推理后执行的步数 |
| `kl_weight` | 10.0 | ACT 的 KL 损失权重 |
| `checkpoint_interval` | 5000 | 定期保存间隔 |
| `best_check_interval` | 100 | 检查是否更新 best 的间隔 |
| `num_workers` | 4 | 数据加载进程数 |
| `tracker` | none | 不连接实验跟踪服务 |

日志应显示 `Using device: cuda`。首次运行可能下载 ResNet18 预训练权重。只想验证流程时，把名称改为 `--name act_easy_smoke`，并加 `--steps 100 --batch-size 8`；短训练不用于判断最终效果。

**LeRobot 导入兼容：** 如果旧脚本报 `cannot import name 'dataset_to_policy_features' from 'lerobot.datasets.utils'`，原因是函数迁移到了 `lerobot.utils.feature_utils`。当前 ACT 与 Diffusion 脚本已使用新路径并保留旧版回退，无需因此重装环境。

### 11.3 输出、续训与评估

```text
outputs/act_easy_v1/
├── model.safetensors       最终模型
├── config.json             模型配置
├── train_config.json       训练结束后写入的训练配置
├── train_state.pt          优化器状态和步数
├── best/                   检查时单批训练 loss 最低的模型
└── checkpoint_5000/        定期检查点（后续还有 checkpoint_10000 等）
```

`best/` 不等于最高任务成功率，需要仿真评估。使用新的 `--name` 区分实验，避免覆盖已有结果。续训时沿用原模型配置，并指定实际存在的检查点：

```bash
python scripts/train/train_act.py \
  --dataset data/recordings/easy_task_v1 \
  --config configs/policy/act/baseline.yaml \
  --name act_easy_v1 \
  --resume outputs/act_easy_v1/checkpoint_10000 \
  --steps 60000 \
  --num-workers 4 \
  --best-check-interval 100 \
  --tracker none
```

`--steps 60000` 是目标总步数。脚本恢复模型、优化器与步数，不恢复随机数状态或数据迭代位置；本次运行的 best loss 也会重新累计。

```bash
conda activate isaac5.1-so-arm101
python scripts/eval/eval_act_policy.py \
  --checkpoint outputs/act_easy_v1/best \
  --env figure_shape_placement_easy \
  --episodes 20 --max-steps 500 --gui
```

去掉 `--gui` 默认无窗口运行。Medium 数据对应 `--env figure_shape_placement`。`outputs/smolvla_iql_easy_v1` 是 SmolVLA 权重，不能作为 ACT 的续训检查点。

#### 多检查点批量评估

训练完成后，使用单环境评估脚本依次比较中间检查点、最低训练 loss 模型和最终模型。所有模型使用相同种子，便于在一致的随机场景序列下比较：

```bash
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab

python scripts/eval/sweep_act_eval_single.py \
  --checkpoint outputs/act_easy_v1 \
  --checkpoints 30000 40000 50000 55000 \
  --best \
  --latest \
  --env figure_shape_placement_easy \
  --episodes 50 \
  --max-steps 500 \
  --seed 42
```

- `--checkpoints 30000 40000 50000 55000`：评估对应的四个 `checkpoint_<步数>/` 目录。
- `--best`：额外评估 `best/`，它按训练 loss 选择，不保证任务成功率最高。
- `--latest`：额外评估根目录中的最终模型；本次为训练完成 60000 步的模型，不需要 `checkpoint_60000/`。
- `--episodes 50`：每个模型评估 50 次；上述六个模型合计计划评估 300 次。
- `--max-steps 500`：每条 episode 最多执行 500 步。
- `--seed 42`：固定评估随机种子，不表示只能评估训练时的种子。

脚本默认无窗口，依次调用 `eval_act_policy.py`，汇总成功率、平均步数等结果，并保存评估记录。筛选出表现好的模型后，应再用 `--seed 50`、`--seed 123` 等未参与选型的场景评估；保持同一检查点和其他参数不变，避免只依据种子 42 的结果判断泛化能力。

### 11.4 可视化数据集（可以使用现有 Isaac 环境）

最简单的方法是直接用视频播放器打开 `videos/` 下的 MP4。需要同步查看双相机画面、关节状态、目标动作及奖励时，使用项目的 Rerun 脚本。

**新开终端**，不会停止原终端的训练。此前检查发现 Isaac 环境已有 OpenCV、NumPy、pandas、pyarrow、Pillow，但缺少 `rerun-sdk`；这是检查时的状态，不代表已安装完成。

```bash
conda activate isaac5.1-so-arm101
python -m pip install "numpy==1.26.0" "opencv-python==4.11.0.86" "rerun-sdk==0.22.1"

# 先查看第一条；小缓存与降采样减少可视化资源占用
python scripts/visualize_lerobot_rerun.py \
  data/recordings/easy_task_v1 \
  --episodes 0 --batch-size 100 --downsample 3

# 查看前三条的全部帧
python scripts/visualize_lerobot_rerun.py \
  data/recordings/easy_task_v1 \
  --episodes 0 1 2 --batch-size 100
```

可视化不启动 Isaac Sim，也不修改数据。窗口需要桌面显示环境；使用底部时间轴播放或拖动。`--episodes` 的编号从 0 开始，**用空格分隔，不用逗号**。

| 参数 | 功能 |
|---|---|
| `--max-episodes 10` | 未指定 episodes 时，默认最多加载 10 条 |
| `--skip-images` | 跳过图像，仅查看状态、动作、奖励等 |
| `--downsample 3` | 每 3 帧显示一帧 |
| `--batch-size 100` | 视频预读缓存大小，与训练 batch size 无关 |
| `--save /tmp/easy_episode0.rrd` | 保存 Rerun 文件，不自动打开窗口 |

保存的文件可用 `rerun /tmp/easy_episode0.rrd` 打开。

**依赖隔离选项：** 安装 Rerun 可能调整 Isaac 环境中的共享依赖版本。如果安装提示冲突，使用独立环境，不要强行升级仿真依赖：

```bash
python3 -m venv venvs/rerun
venvs/rerun/bin/python -m pip install rerun-sdk opencv-python numpy pandas pyarrow pillow
venvs/rerun/bin/python scripts/visualize_lerobot_rerun.py \
  data/recordings/easy_task_v1 --episodes 0 --batch-size 100
```


## 12. 整个项目的训练程序顺序

参见 [训练流程与程序顺序（中文）](docs/workflow/training_pipeline_cn.md)：包含完整流程图、各入口的运行环境和输入输出、ACT 主线、SmolVLA/IQL/PPO 与独立 SAC 分支。主要入口脚本已添加中文流程及关键逻辑注释。

当前 ACT 主线：`download_demo.py`（或 `record_episodes.py`）→ `verify_lerobot_dataset.py` / `visualize_lerobot_rerun.py` → `train_act.py` → `eval_act_policy.py` → 多检查点评估。ACT、Diffusion、SmolVLA 是不同分支，无需依次训练。
