# SO-101 Lab — Personal Notes

> Personalized quick-start guide for **this machine**. For the full upstream project docs (architecture, every script, results tables, attribution), see [docs/](docs/) — this file trades completeness for "what do I actually type."

---

## 1. What this project is

Train manipulation policies for the SO-101 arm entirely in simulation, then optionally deploy to the real robot:

```
teleoperate in Isaac Sim  →  record LeRobot-format dataset  →  train BC policy (ACT / Diffusion / SmolVLA)
                                                              →  eval in sim
                                                              →  (optional) RL fine-tune (IQL offline / PPO online)
                                                              →  deploy on real SO-101
```

Task scene: a children's shape-sorting toy (cube, star, hexagon, cylinder, ...) on a workbench. The robot has to pick a shape and place it in a target zone. Two difficulty presets exist as separate task envs:
- `figure_shape_placement_easy` — small spawn zone, start here
- `figure_shape_placement` — 4x spawn area, harder

Reference success rates from the upstream project (SmolVLA + IQL): 86–88% on easy, 48% on medium. Plain BC (ACT) alone: ~56%.

---

## 2. Environments on this machine

The project needs **two incompatible Python environments** because Isaac Sim pins Python 3.11 and LeRobot requires Python 3.12. Tool venvs are optional extras. **Both are set up and working on this machine.**

| Role | What's used on this machine | Python | Status |
|---|---|---|---|
| **isaaclab-env** (sim, teleop, RL env loop, eval) | conda env **`isaac5.1-so-arm101`** | 3.11.15 | ✅ Isaac Sim 5.1.0.0 + Isaac Lab 2.3.0 + torch 2.7.0+cu128 (already existed, reused as-is) |
| **lerobot-env** (ACT/Diffusion/SmolVLA training, IQL critics, dataset download) | conda env **`lerobot-env`** | 3.12.14 | ✅ created fresh: LeRobot 0.6.2 + torch 2.11.0+cu128 + so101-lab |
| `venvs/rerun` | dataset visualization | any | optional, not created |
| `venvs/viewer` | live camera preview during recording | any | optional, not created |

**Note:** `uv` is *not* required anywhere. Both envs were built with plain `conda create` + `pip`. Nothing in `pyproject.toml` depends on uv.

**Note:** `activate_isaaclab.sh` / `activate_lerobot.sh` assume a venv with `bin/activate`, which conda envs don't have. Don't use them — just `conda activate <env-name>` directly.

### Activating

```bash
# isaaclab-env — sim, teleop, recording, eval
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab

# lerobot-env — BC/VLA training, dataset download/inspection
conda activate lerobot-env
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
```

---

## 3. Setup already done

### isaaclab-env (`isaac5.1-so-arm101`)

First run of `record_episodes.py` was missing packages — installed:
```bash
conda activate isaac5.1-so-arm101
pip install -e .
pip install pyarrow av datasets huggingface_hub accelerate num2words safetensors \
            cma nvidia-ml-py pyserial deepdiff feetech-servo-sdk diffusers
pip install "click==8.1.7"   # pinned — see gotcha below
```

**Gotcha: click version pin.** Installing the packages above upgrades `click` past `8.1.7`, which conflicts with `isaacsim-kernel`'s exact pin (`click==8.1.7`) and can break Isaac Sim. Pinned back to `8.1.7`. **Side effect:** `typer` (and therefore `trackio`) cannot be imported in this env — matches the upstream-documented limitation that trackio only works in lerobot-env. Always use `--tracker wandb` or `--tracker none` here, never `--tracker trackio`.

### lerobot-env

Ubuntu 22.04's apt repos don't ship Python 3.12 (would need the deadsnakes PPA), so this env was built with **conda** instead of a plain venv — simpler, same result:
```bash
conda create -n lerobot-env python=3.12 -y
conda activate lerobot-env
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

git clone https://github.com/huggingface/lerobot.git ~/lerobot
cd ~/lerobot
pip install -e ".[dataset]"      # the `[dataset]` extra is required — plain `pip install -e .`
                                  # is not enough; `lerobot.datasets.lerobot_dataset` raises
                                  # ImportError without it ("datasets" package missing)

cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
pip install -e ".[hardware]"
pip install trackio nvidia-ml-py
```

**Gotcha: SOCKS proxy breaks Hugging Face downloads.** This machine has `ALL_PROXY=socks://127.0.0.1:7897` set globally. `httpx` (used by `huggingface_hub`) only accepts `socks5://`/`socks5h://` proxy schemes, not the bare `socks://` scheme — so any Hub download fails with `ValueError: Unknown scheme for proxy URL`. `HTTPS_PROXY=http://127.0.0.1:7897` (also set) works fine on its own. Fix: drop `ALL_PROXY`/`all_proxy` for the command:
```bash
env -u ALL_PROXY -u all_proxy python scripts/tools/download_demo.py --dataset easy --checkpoint easy
```
(Permanent fix would be changing the system's `ALL_PROXY` to `socks5://127.0.0.1:7897`, not done here.)

### Hardware check
GPU: RTX 4090, 24 GB VRAM — comfortably above the 8 GB minimum. `cmake` is the apt version (required, not snap).

---

## 4. Official demo dataset (already downloaded — no need to record from scratch to get started)

Upstream publishes a pre-recorded dataset + a trained checkpoint on Hugging Face. Already pulled down on this machine:

| Asset | HF repo | Local path |
|---|---|---|
| Easy dataset (200 episodes) | `MSSerg/so101-easy-task-v1` | `data/recordings/easy_task_v1` ✅ downloaded |
| SmolVLA + IQL checkpoint (86–88% SR) | `MSSerg/so101-smolvla-iql-easy-v1` | `outputs/smolvla_iql_easy_v1` ✅ downloaded |
| Medium dataset (300 episodes) | `MSSerg/so101-figure-shape-placement-v1` | not downloaded — `python scripts/tools/download_demo.py --dataset medium` |

Use it directly instead of recording your own data first:
```bash
conda activate lerobot-env
python scripts/train/train_act.py --dataset data/recordings/easy_task_v1 --name my_act_v1

# or eval the downloaded checkpoint (two terminals)
# terminal 1 (lerobot-env):
python scripts/eval/smolvla_server.py --checkpoint outputs/smolvla_iql_easy_v1
# terminal 2 (isaaclab-env):
conda activate isaac5.1-so-arm101
python scripts/eval/eval_vla_policy.py --checkpoint outputs/smolvla_iql_easy_v1 \
    --env figure_shape_placement_easy --episodes 10 --no-domain-rand --gui
```

---

## 5. Day-to-day workflow

### Record a dataset (isaaclab-env)
```bash
conda activate isaac5.1-so-arm101

# keyboard (no physical robot needed) — do NOT use --headless, keyboard capture needs the Isaac Sim window
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=keyboard

# with the real SO-101 leader arm — see §7 below for the full hardware walkthrough
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader --calibration-file=leader_1.json --port=/dev/ttyACM0 \
    --reward-mode success
```

### Train a BC policy (lerobot-env)
```bash
conda activate lerobot-env
python scripts/train/train_act.py \
    --dataset data/recordings/my_task_v1 \
    --config configs/policy/act/baseline.yaml \
    --name act_v1 --tracker wandb
```

### Evaluate in sim (isaaclab-env)
```bash
conda activate isaac5.1-so-arm101
python scripts/eval/eval_act_policy.py \
    --checkpoint outputs/act_v1 \
    --env figure_shape_placement_easy \
    --episodes 20 --max-steps 500 --headless
```

### RL fine-tuning (optional, bigger success-rate jump)
See [docs/workflow/04_offline_rl.md](docs/workflow/04_offline_rl.md) (IQL, recommended first) and [docs/workflow/05_online_rl.md](docs/workflow/05_online_rl.md) (PPO).

---

## 6. USD → STL conversion (for 3D printing real props)

The task props (`assets/props/*.usd` — cube, star, hexagon, cylinder, etc.) only ship as USD; no STL originals exist (unlike the robot itself, which already has STL meshes at `assets/robots/so101/meshes/*.stl`). Wrote `scripts/tools/usd_to_stl.py` to convert them — bakes each mesh's world transform and converts USD stage units to meters (default `--units m`). STL stores no units: import as meters with scale 1. For 3D printing, use `--units mm`, preferably with a separate `assets/props_stl_mm` output directory.

```bash
conda activate isaac5.1-so-arm101
python scripts/tools/usd_to_stl.py assets/props --output assets/props_stl
```

Physical sizes below are shown in millimeters; default STL coordinates are in meters (for example, the cube is `0.04 × 0.04 × 0.0165`):

| Prop | Size (mm) | Printable? |
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
| platfrom_full (target tray) | 200 × 200 × 9 | ✅ (desktop printer bed size) |
| full_game (combined scene) | 50000 × 50000 × 23.4 | ⚠️ includes a giant background plane, not a real prop — don't print |
| workbench | 900 × 600 × 750 | ⚠️ real-furniture scale, reference only |

---

## 7. Real leader-arm hardware teleop

Confirmed working end-to-end on this machine: `so101leader` device reads joint positions directly off the physical leader arm (torque disabled, free-moving) and maps them 1:1 to the sim robot's joints — no IK, direct joint mimicry. Only the leader arm needs to be connected (the sim IS the follower).

### After connecting: identify the serial port

```bash
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
python -m serial.tools.list_ports -v
ls -l /dev/serial/by-id/
```

If several devices are listed, unplug and reconnect the leader USB and compare the results. The device number can change, and the port may be named `ttyUSB0` instead of `ttyACM0`. The legacy error message suggests `python lerobot/find_port.py`, but that path does not exist in this project; use the commands above.

This machine previously detected serial number `5A7A057348`. Prefer its persistent device link over a numbered port; replace it with your own device link on other hardware:

```bash
export LEADER_PORT=/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7A057348-if00
ls -l "$LEADER_PORT"
```

Run the following commands in the same terminal; set `LEADER_PORT` again in a new terminal. A persistent link still requires a connected USB device and cannot fix a disconnect.

### Diagnose connection and permission errors

| Error | Meaning and fix |
|---|---|
| `No such file or directory` / `Errno 2` | Device is absent. List ports again, check the USB data cable, power and connection, then reconnect. `chmod` cannot fix an absent device. |
| `Permission denied` / `Errno 13` | Device exists but the process lacks permission. Check `dialout` membership below. |
| `python: can't open file ... lerobot/find_port.py` | Tool path is absent; use `python -m serial.tools.list_ports -v`. |

Use `ls -l "$LEADER_PORT"` to inspect the link and `ls -lL "$LEADER_PORT"` for device permissions. Serial devices typically belong to `root:dialout` with mode 660.

```bash
id -nG          # groups active in this shell
id -nG "$USER"  # account group membership
# Only if the account is not already in dialout:
sudo usermod -aG dialout "$USER"
# If membership is configured but not active in this shell:
newgrp dialout
conda activate isaac5.1-so-arm101
```

Alternatively, log out of the desktop session and log in again. **Opening another terminal alone may not apply the new group**, because it can inherit the old desktop session's groups. There is no need to grant everyone serial access using `chmod 666`.

### One-time: calibrate

For the same leader with a valid `leader_1.json`, reconnecting USB or changing port numbers normally does not require recalibration. Proceed directly to control testing. Run calibration only for initial setup or when recalibration is needed; move gently through the normal mechanical range.

```bash
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
python scripts/calibrate_leader.py --port "$LEADER_PORT" --name leader_1
```
Move to middle of range → Enter, move every joint through its full range → Enter. Saves to `so101_lab/devices/lerobot/.cache/leader_1.json`.

### Test control (no recording)
```bash
python scripts/teleop/teleop_agent.py \
    --task figure_shape_placement_easy \
    --teleop-device so101leader \
    --port "$LEADER_PORT" \
    --calibration-file leader_1.json \
    --episode-length 300
```
Focus the Isaac Sim window, press `Space` to start teleoperation, then move the leader slowly to check tracking.

**Gotcha 1:** `teleop_agent.py` uses `--task`, not `--env` — inconsistent with `record_episodes.py`, which uses `--env`. Easy to typo between the two scripts.

**Gotcha 2:** `teleop_agent.py`'s `--episode-length` **defaults to 25 seconds** — meant for quick smoke-testing, not enough time to actually complete a pick-and-place while still learning the controls. The env resets on a plain timeout (`env.episode_length_buf[0] >= env.max_episode_length`), not because it detected failure. Pass a bigger value (`--episode-length 120` or more) while practicing.

### Record for real
```bash
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader \
    --calibration-file=leader_1.json \
    --port="$LEADER_PORT" \
    --reward-mode success
```
(`record_episodes.py` already defaults to a 120s episode length, unlike `teleop_agent.py`.)

---

## 8. Keyboard teleop controls

Recording script prints this table at startup. Control is **Cartesian delta of the gripper** (not raw joint angles) for most keys — `A`/`D` and `U`/`O` are direct joint control.

| Key | Action |
|---|---|
| `Space` | start teleop (**do this first**, nothing else works until you do) |
| `N` | start recording this episode |
| `F` | finish + save episode |
| `R` | restart (reset, same scene) |
| `X` | discard episode, reset |
| `T` | debug info |
| `Escape` | quit |
| `W` / `S` | gripper forward / backward |
| `A` / `D` | base rotate left / right (shoulder_pan joint) |
| `Q` / `E` | gripper up / down |
| `I` / `K` | pitch up / down |
| `J` / `L` | yaw left / right |
| `U` / `O` | gripper open / close |

Keys are **hold-to-move, release-to-stop** (not click-to-nudge). Tune speed with `--sensitivity <float>` (default 1.0). Window focus must be on the Isaac Sim viewport, not the terminal.

Typical loop: `Space` → move around to warm up → `X`/`R` to reset if needed → `N` → perform the pick-and-place → `F` to save (or `X` to discard) → repeat 50–200x → `Escape`.

---

## 9. Known issues on this setup

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyarrow'` (or `av`, `datasets`) in isaaclab-env | isaaclab-env additional packages step wasn't done | installed, see §3 |
| `[WARNING] Viewer venv not found` | optional `venvs/viewer` tool env not created | harmless, ignore, or `python3 -m venv venvs/viewer && source venvs/viewer/bin/activate && pip install opencv-python numpy` |
| `trackio`/`typer` import fails in isaaclab-env | `click` pinned to 8.1.7 for Isaac Sim compatibility | expected — use `--tracker wandb`/`none` in isaaclab-env, `--tracker trackio` only in lerobot-env |
| Keyboard controls don't respond | ran with `--headless`, no window to receive key events; or forgot `Space` first | drop `--headless` for keyboard teleop; press `Space` before other keys |
| `ModuleNotFoundError: No module named 'lerobot'` | ran a lerobot-only script (e.g. `download_demo.py`, `train_act.py`) inside isaaclab-env | switch to `conda activate lerobot-env` |
| `ImportError: 'datasets' is required` when importing `lerobot.datasets...` | installed lerobot with plain `pip install -e .` instead of the `[dataset]` extra | `pip install -e ".[dataset]"` from the lerobot repo root |
| `ValueError: Unknown scheme for proxy URL 'socks://...'` on any HF download | system `ALL_PROXY` uses bare `socks://`, httpx only accepts `socks5://`/`socks5h://` | `env -u ALL_PROXY -u all_proxy <command>` |
| `PermissionError: [Errno 13] Permission denied: '/dev/ttyACM0'` | terminal session predates being added to the `dialout` group | `newgrp dialout` or log out and back in (a new terminal alone may not suffice) |
| `teleop_agent.py: error: unrecognized arguments: --env` | script takes `--task`, not `--env` (unlike `record_episodes.py`) | use `--task <env_name>` |
| Sim resets mid pick-and-place in `teleop_agent.py` | `--episode-length` defaults to 25s (smoke-test default), plain timeout, not a failure detection | pass `--episode-length 120` or more |

For anything not covered here: [docs/troubleshooting.md](docs/troubleshooting.md).

---

## 10. Where to go deeper

| Need | File |
|---|---|
| Task-oriented how-to ("I want to record / train / eval / deploy") | [docs/guide.md](docs/guide.md) |
| Full step-by-step workflow | [docs/workflow/](docs/workflow/) |
| Why 4 venvs, data format, timing | [docs/architecture.md](docs/architecture.md) |
| Every script's flags | [docs/reference/index.md](docs/reference/index.md) |
| Experiment results detail | [docs/results/](docs/results/) |

---

## 11. Download datasets, train ACT, and visualize demonstrations

Run all commands from the project root:
```bash
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
```

### 11.1 Dataset locations and downloads

| Dataset | Hugging Face repository | Local directory | Evaluation environment |
|---|---|---|---|
| Easy | `MSSerg/so101-easy-task-v1` | `data/recordings/easy_task_v1` | `figure_shape_placement_easy` |
| Medium | `MSSerg/so101-figure-shape-placement-v1` | `data/recordings/figure_shape_placement_v1` | `figure_shape_placement` |

The local Easy metadata reports 200 episodes, 74,826 frames, and 30 FPS. Its full path is `/home/cym/ROS/Arms/isaac_so_arm101/so101-lab/data/recordings/easy_task_v1`.

```bash
conda activate lerobot-env
# Use medium for the harder dataset, or all for both datasets.
# Remove ALL_PROXY to avoid this machine's socks:// compatibility issue.
env -u ALL_PROXY -u all_proxy python scripts/tools/download_demo.py --dataset easy

# Check structure, metadata, LeRobot loading, and the first sample.
python scripts/verify_lerobot_dataset.py data/recordings/easy_task_v1
```

The downloader skips any nonempty destination directory. It **does not verify completeness or repair partial downloads**. Check existing data before training.

```text
easy_task_v1/
├── meta/       Dataset information, statistics, tasks, and episode indices
├── data/       Parquet: joint states, actions, rewards, etc.
└── videos/
    ├── observation.images.top/chunk-000/file-000.mp4
    └── observation.images.wrist/chunk-000/file-000.mp4
```

An MP4 can contain several episodes; its file number is not an episode index.

### 11.2 ACT training and a short smoke run

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

CLI arguments override YAML values, which override script defaults. The effective settings above are:

| Parameter | Value | Meaning |
|---|---|---|
| `steps` | 60000 | Total optimizer updates, not epochs |
| `batch_size` | 24 | Samples per batch; add `--batch-size 8` if GPU memory runs out |
| `lr` | `1e-5` | Learning rate |
| `chunk_size` / `n_action_steps` | 25 / 25 | Predicted action sequence length / actions executed per inference |
| `kl_weight` | 10.0 | ACT KL loss weight |
| `checkpoint_interval` | 5000 | Periodic checkpoint interval |
| `best_check_interval` | 100 | Interval for checking whether to replace best |
| `num_workers` | 4 | Data loader processes |
| `tracker` | none | No experiment tracking service |

Expect `Using device: cuda`. The first run may download pretrained ResNet18 weights. For a smoke run, change the name to `--name act_easy_smoke` and add `--steps 100 --batch-size 8`. This checks the training pipeline, not policy quality.

**LeRobot import compatibility:** Newer versions moved `dataset_to_policy_features` from `lerobot.datasets.utils` to `lerobot.utils.feature_utils`. The ACT and Diffusion scripts now use the new location with a fallback for older versions. An environment reinstall is not needed for this import error.

### 11.3 Outputs, resuming, and evaluation

```text
outputs/act_easy_v1/
├── model.safetensors       Final model
├── config.json             Model configuration
├── train_config.json       Training configuration written at completion
├── train_state.pt          Optimizer state and step
├── best/                   Lowest checked single-batch training loss
└── checkpoint_5000/        Periodic checkpoint (then checkpoint_10000, etc.)
```

`best/` is selected by training loss, not task success rate. Evaluate in simulation to measure performance. Use different run names to avoid overwriting experiments. Resume from an existing checkpoint with matching dataset and model settings:

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

`--steps` is the target total, including previous training. Resume restores the model, optimizer, and step, but not RNG state or data iterator position. Best-loss tracking also restarts for the resumed run.

```bash
conda activate isaac5.1-so-arm101
python scripts/eval/eval_act_policy.py \
  --checkpoint outputs/act_easy_v1/best \
  --env figure_shape_placement_easy \
  --episodes 20 --max-steps 500 --gui
```

Omit `--gui` for headless evaluation. Use `--env figure_shape_placement` for Medium. The downloaded `outputs/smolvla_iql_easy_v1` contains SmolVLA weights and cannot serve as an ACT resume checkpoint.

#### Compare multiple checkpoints

After training, use the single-environment sweep to compare intermediate checkpoints, the lowest-training-loss model, and the final model. A shared seed provides a consistent randomized scenario sequence across models:

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

- `--checkpoints 30000 40000 50000 55000`: evaluate the four corresponding `checkpoint_<step>/` directories.
- `--best`: also evaluate `best/`, selected by training loss rather than task success rate.
- `--latest`: also evaluate the final model in the run root; this run finished at 60000 steps, so no `checkpoint_60000/` directory is required.
- `--episodes 50`: run 50 episodes per model, for 300 planned episodes across these six models.
- `--max-steps 500`: allow at most 500 steps per episode.
- `--seed 42`: fix the evaluation seed; evaluation does not have to use the training seed.

The sweep runs headlessly, invokes `eval_act_policy.py` sequentially, summarizes success rates and episode lengths, and saves evaluation records. After selecting a model, evaluate it on additional seeds such as `--seed 50` and `--seed 123` that were not used for selection. Keep the checkpoint and other settings fixed to assess generalization beyond seed 42.

### 11.4 Visualize data using the existing Isaac environment

For a quick look, open an MP4 under `videos/` in a video player. The project's Rerun script synchronizes both camera streams with joint states, target actions, and rewards.

Open a **new terminal** so the training process keeps running. At the previous dependency check, the Isaac environment had OpenCV, NumPy, pandas, pyarrow, and Pillow, but lacked `rerun-sdk`; this records that check, not a completed installation.

```bash
conda activate isaac5.1-so-arm101
python -m pip install "numpy==1.26.0" "opencv-python==4.11.0.86" "rerun-sdk==0.22.1"

# First episode, with a small video cache and frame downsampling.
python scripts/visualize_lerobot_rerun.py \
  data/recordings/easy_task_v1 \
  --episodes 0 --batch-size 100 --downsample 3

# All frames from the first three episodes.
python scripts/visualize_lerobot_rerun.py \
  data/recordings/easy_task_v1 \
  --episodes 0 1 2 --batch-size 100
```

This does not start Isaac Sim or modify the dataset. The viewer needs a graphical desktop. Play or scrub using the bottom timeline. Episode indices start at zero and are **space-separated, not comma-separated**.

| Option | Purpose |
|---|---|
| `--max-episodes 10` | Default limit when explicit episodes are not provided |
| `--skip-images` | Skip images and inspect states, actions, rewards, etc. |
| `--downsample 3` | Display every third frame |
| `--batch-size 100` | Video prefetch cache size, unrelated to training batch size |
| `--save /tmp/easy_episode0.rrd` | Save a Rerun recording instead of spawning a viewer |

Open saved recordings with `rerun /tmp/easy_episode0.rrd`.

**Optional dependency isolation:** Installing Rerun may change shared dependency versions in the Isaac environment. If installation reports conflicts, use a separate environment instead of forcing simulator dependency upgrades:

```bash
python3 -m venv venvs/rerun
venvs/rerun/bin/python -m pip install rerun-sdk opencv-python numpy pandas pyarrow pillow
venvs/rerun/bin/python scripts/visualize_lerobot_rerun.py \
  data/recordings/easy_task_v1 --episodes 0 --batch-size 100
```


## 12. Training program order

See the [Chinese training pipeline guide](docs/workflow/training_pipeline_cn.md) for the complete diagram, script inputs/outputs, environments, ACT baseline, SmolVLA/IQL/PPO, and the separate SAC branch. Main entry scripts now include Chinese workflow and key-stage comments.

ACT path: `download_demo.py` (or `record_episodes.py`) → `verify_lerobot_dataset.py` / `visualize_lerobot_rerun.py` → `train_act.py` → `eval_act_policy.py` → checkpoint comparison. ACT, Diffusion, and SmolVLA are alternative branches, not sequential prerequisites.
