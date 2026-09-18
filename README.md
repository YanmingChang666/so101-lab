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

The project needs **two incompatible Python environments** because Isaac Sim pins Python 3.11 and LeRobot 0.5.1 requires Python 3.12. Tool venvs are optional extras.

| Role | What to use on this machine | Python | Status |
|---|---|---|---|
| **isaaclab-env** (sim, teleop, RL env loop, eval) | conda env **`isaac5.1-so-arm101`** | 3.11.15 | ✅ already has Isaac Sim 5.1.0.0 + Isaac Lab 2.3.0 + torch 2.7.0+cu128 — reused as-is, no need to build a fresh venv |
| **lerobot-env** (ACT/Diffusion/SmolVLA training, IQL critics) | a plain `python3.12 -m venv` (not conda, not uv — none required) | 3.12 | needs to be created, see below |
| `venvs/rerun` | dataset visualization (rerun-sdk) | any | optional |
| `venvs/viewer` | live camera preview during recording | any | optional |

**Note:** `uv` is *not* required anywhere — it's just what the upstream docs use for speed. Plain `pip`/`venv`/`conda` work identically; nothing in `pyproject.toml` depends on uv.

**Note:** `activate_isaaclab.sh` assumes a venv with `bin/activate`, which the conda env doesn't have. Don't use it — just `conda activate isaac5.1-so-arm101` directly.

### Activating

```bash
# isaaclab-env
conda activate isaac5.1-so-arm101
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab

# lerobot-env (once created — see setup below)
source ~/venvs/lerobot-env/bin/activate
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
```

---

## 3. Setup already done in `isaac5.1-so-arm101`

These were missing the first time `record_episodes.py` was run and have been installed:

```bash
conda activate isaac5.1-so-arm101
pip install -e .
pip install pyarrow av datasets huggingface_hub accelerate num2words safetensors \
            cma nvidia-ml-py pyserial deepdiff feetech-servo-sdk diffusers
pip install "click==8.1.7"   # pinned — see gotcha below
```

### Gotcha: click version pin
Installing the packages above upgrades `click` past `8.1.7`, which conflicts with `isaacsim-kernel`'s exact pin (`click==8.1.7`) and can break Isaac Sim. It has been pinned back. **Side effect:** `typer` (and therefore `trackio`) cannot be imported in this env — this matches the upstream-documented limitation that **trackio only works in lerobot-env**. In isaaclab-env, always pass `--tracker wandb` or `--tracker none` to scripts, never `--tracker trackio`.

### lerobot-env — not yet created
```bash
python3.12 -m venv ~/venvs/lerobot-env
source ~/venvs/lerobot-env/bin/activate
pip install --upgrade pip
git clone https://github.com/huggingface/lerobot.git ~/lerobot   # need 0.5.1+
cd ~/lerobot && pip install -e .
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd /home/cym/ROS/Arms/isaac_so_arm101/so101-lab
pip install -e ".[hardware]"
pip install trackio nvidia-ml-py
```

Then fill in `.env` (copy from `.env.example`) with:
```
LEROBOT_ENV=/home/cym/venvs/lerobot-env
LEROBOT_SRC=/home/cym/lerobot/src
```
(`ISAACLAB_ENV` isn't usable with conda — just `conda activate isaac5.1-so-arm101` manually instead of using `act-isaac`.)

### Hardware check
GPU: RTX 4090, 24 GB VRAM — comfortably above the 8 GB minimum. `cmake` is the apt version (required, not snap).

---

## 4. Day-to-day workflow

### Record a dataset (isaaclab-env)
```bash
conda activate isaac5.1-so-arm101

# keyboard (no physical robot needed) — do NOT use --headless, keyboard capture needs the Isaac Sim window
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=keyboard

# with a real SO-101 leader arm
python scripts/teleop/record_episodes.py \
    --output data/recordings/my_task_v1 \
    --env figure_shape_placement_easy \
    --teleop-device=so101leader --calibration-file=leader_1.json --port=/dev/ttyACM0 \
    --reward-mode success
```

Or skip recording and grab the official demo dataset + checkpoint:
```bash
python scripts/tools/download_demo.py --dataset easy --checkpoint easy
```

### Train a BC policy (lerobot-env)
```bash
source ~/venvs/lerobot-env/bin/activate
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

## 5. Keyboard teleop controls

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

## 6. Known issues on this setup

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'pyarrow'` (or `av`, `datasets`) | isaaclab-env additional packages step (setup doc §3) wasn't done | installed, see §3 above |
| `[WARNING] Viewer venv not found` | `venvs/viewer` optional tool env not created | harmless, ignore, or `python3 -m venv venvs/viewer && source venvs/viewer/bin/activate && pip install opencv-python numpy` |
| `trackio`/`typer` import fails in isaaclab-env | `click` pinned to 8.1.7 for Isaac Sim compatibility | expected — use `--tracker wandb`/`none` in isaaclab-env, `--tracker trackio` only in lerobot-env |
| Keyboard controls don't respond | ran with `--headless`, no window to receive key events; or forgot `Space` first | drop `--headless` for keyboard teleop; press `Space` before other keys |

For anything not covered here: [docs/troubleshooting.md](docs/troubleshooting.md).

---

## 7. Where to go deeper

| Need | File |
|---|---|
| Task-oriented how-to ("I want to record / train / eval / deploy") | [docs/guide.md](docs/guide.md) |
| Full step-by-step workflow | [docs/workflow/](docs/workflow/) |
| Why 4 venvs, data format, timing | [docs/architecture.md](docs/architecture.md) |
| Every script's flags | [docs/reference/index.md](docs/reference/index.md) |
| Experiment results detail | [docs/results/](docs/results/) |
