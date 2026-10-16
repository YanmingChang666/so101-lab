#!/usr/bin/env python3
"""Prepare LeRobot dataset with rewards.

Computes `next.reward` using classifier, VIP, or both. Optionally resizes videos.
Used for SAC training, IQL critics, weighted BC, etc.

Run in lerobot-env:
    eval "$(./activate_lerobot.sh)"

    # Classifier only (default)
    python scripts/train/prepare_reward_dataset.py \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_clf \
        --reward-model outputs/reward_classifier_v1/best

    # VIP only
    python scripts/train/prepare_reward_dataset.py \
        --reward-source vip \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_vip \
        --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
        --vip-label-dataset data/recordings/figure_shape_placement_v4_labeled \
        --vip-use-labeled --vip-goal-mode min

    # Composite (classifier + VIP)
    python scripts/train/prepare_reward_dataset.py \
        --reward-source composite \
        --dataset data/recordings/figure_shape_placement_v4 \
        --output data/recordings/figure_shape_placement_v4_composite \
        --reward-model outputs/reward_classifier_v1/best \
        --vip-goal-dataset data/recordings/figure_shape_placement_v4 \
        --w-vip 1.0 --image-size 128
"""

# 中文流程说明（完整顺序见 docs/workflow/training_pipeline_cn.md）
# 定位：离线强化学习第一步：为示范补充视觉奖励。
# 运行环境与输入：lerobot-env；输入原始数据、输出目录，以及 VIP 目标数据或分类器权重。
# 输出与下游：在输出数据集的 next.reward 列写入 VIP、分类器或组合奖励。
# 关键逻辑与边界：这一步训练的是零个策略网络，只为后续 critics 准备监督信号。
#     使用与原始目录不同的新输出目录；已有奖励的复制分支会删除已存在的输出目录。
#     patch_total_tasks 还可能补写源数据元信息，不能认为整个操作完全只读。

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

# 定位本仓库根目录并加入模块搜索路径，使直接执行脚本时能导入 so101_lab。
# Add so101_lab to path when running from lerobot-env
SO101_LAB_ROOT = Path(__file__).resolve().parent.parent.parent
if str(SO101_LAB_ROOT) not in sys.path:
    sys.path.insert(0, str(SO101_LAB_ROOT))

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from lerobot.datasets.dataset_tools import add_features
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from so101_lab.rewards.classifier import RewardClassifier


def _sample_to_hwc_uint8(sample, key: str) -> np.ndarray:
    """将 LeRobot 单帧图像转换为奖励模型使用的 HWC / uint8 格式。

    参数：sample 为 dataset[i] 返回的字典，key 为相机特征名。
    兼容 observation.image.* 与 observation.images.* 两种键名。
    常见输入是 CPU 上的 (3,H,W)、float32、[0,1] 张量；输出为
    (H,W,3)、uint8、[0,255] 数组，不改变 RGB 通道顺序。

    注意：此函数只处理代码中列出的常见情况，并非通用图像转换器。
    Tensor.numpy() 要求张量在 CPU 且无需梯度；浮点转换仅针对 float32，
    不会裁剪越界像素，也不会自动把 GPU 张量移回 CPU。
    找不到原键及兼容键时抛出 KeyError，不静默跳过这一帧。"""
    # 先使用指定相机键；缺失时才尝试单数 image / 复数 images 的兼容写法。
    img = sample.get(key)
    if img is None:
        alt_key = key.replace("observation.image.", "observation.images.")
        if alt_key == key:
            alt_key = key.replace("observation.images.", "observation.image.")
        img = sample.get(alt_key)
    if img is None:
        raise KeyError(f"Missing image key: {key}")

    if isinstance(img, torch.Tensor):
        img = img.numpy()
    # 通过首维是否为 3 判断 CHW，这是针对常见 RGB 图像的布局约定。
    if img.shape[0] == 3:  # CHW -> HWC
        img = np.transpose(img, (1, 2, 0))
    # 假设 float32 像素已归一化到 [0,1]，恢复为奖励接口所需的 8 位像素。
    if img.dtype == np.float32:
        img = (img * 255).astype(np.uint8)
    return img


def compute_classifier_rewards(
    dataset: LeRobotDataset,
    reward_model: RewardClassifier,
) -> np.ndarray:
    """使用已训练的成功分类器，为每个数据帧计算二值奖励。

    根据 reward_model.image_keys 取出分类器需要的所有相机图像，
    转换格式后调用 predict_reward。分类器内部按其 threshold 把
    成功概率转为 0/1；这里不训练分类器，也不修改数据集。
    返回长度为 len(dataset) 的 float32 数组，位置 i 对应 dataset[i]。
    日志统计的是成功“帧”比例，不是任务 episode 成功率。"""
    total_frames = len(dataset)
    # 建立与 dataset 顺序相同的奖励数组；不同来源以加法合并，不累加旧 next.reward。
    rewards = np.zeros(total_frames, dtype=np.float32)
    # 相机列表来自分类器检查点配置；不要在这里硬编码只使用 top 或 wrist。
    image_keys = reward_model.image_keys

    print(f"Computing classifier rewards for {total_frames} frames...")
    print(f"  Image keys: {image_keys}")

    for i in tqdm(range(total_frames), desc="Classifier rewards"):
        sample = dataset[i]
        obs = {key: _sample_to_hwc_uint8(sample, key) for key in image_keys}
        rewards[i] = reward_model.predict_reward(obs)

    success_frames = (rewards > 0.5).sum()
    print(f"  Success frames: {success_frames} ({100 * success_frames / total_frames:.1f}%)")
    return rewards


def compute_vip_rewards(
    dataset: LeRobotDataset,
    vip_model,
    image_key: str,
) -> np.ndarray:
    """使用 VIP 视觉表征，为指定相机的每一帧计算密集奖励。

    vip_model 已在初始化时加载视觉编码器并构建目标图像特征。
    本函数只逐帧取图并调用 compute_reward，不反向传播、不训练策略。
    在本脚本的默认模型设置下，奖励是当前特征到目标特征的负距离：
    越接近 0 表示越接近目标，不是 [0,1] 的成功概率。
    返回按数据集顺序排列的一维 float32 数组；VIP 权重在 main 中再乘。"""
    total_frames = len(dataset)
    # 建立与 dataset 顺序相同的奖励数组；不同来源以加法合并，不累加旧 next.reward。
    rewards = np.zeros(total_frames, dtype=np.float32)

    print(f"Computing VIP rewards for {total_frames} frames...")
    print(f"  Image key: {image_key}")

    for i in tqdm(range(total_frames), desc="VIP rewards"):
        sample = dataset[i]
        img = _sample_to_hwc_uint8(sample, image_key)
        # 奖励对应当前帧图像；这里没有额外读取 i+1，也不执行仿真动作。
        rewards[i] = vip_model.compute_reward({image_key: img})

    print(f"  Range: [{rewards.min():.2f}, {rewards.max():.2f}], mean={rewards.mean():.2f}")
    return rewards


def resize_videos(output_path: Path, target_size: int) -> list[str]:
    """把输出数据集下的全部 MP4 重新编码为 target_size × target_size。

    需要系统中可以调用 ffmpeg，并支持 libx264 编码器。
    scale 直接缩放到正方形，不补边、不保持原始长宽比例；例如 640×480
    会被拉伸成 128×128。编码使用 H.264 / yuv420p，去掉音频。
    每个视频先写临时文件，编码成功后才替换对应 MP4；失败时删除临时文件
    并抛出错误。该操作不是整个数据集级别的事务，先前成功的视频不会回滚。
    返回已替换视频相对于输出目录的路径列表；没有视频则返回空列表。"""
    video_dir = output_path / "videos"
    mp4_files = sorted(video_dir.glob("**/*.mp4"))

    if not mp4_files:
        return []

    patched = []
    print(f"\nRe-encoding {len(mp4_files)} videos to {target_size}x{target_size}...")

    for mp4 in tqdm(mp4_files, desc="Resizing videos"):
        # 临时文件放在原视频同一目录，成功后再 rename 替换，避免直接写坏原视频。
        with tempfile.NamedTemporaryFile(suffix=".mp4", dir=mp4.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)

        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(mp4),
                "-vf", f"scale={target_size}:{target_size}",
                # fast 为编码速度预设，CRF 23 控制有损压缩质量；并非无损拷贝。
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-an", str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            tmp_path.unlink(missing_ok=True)
            raise RuntimeError(f"ffmpeg failed for {mp4}: {result.stderr}")

        tmp_path.rename(mp4)
        patched.append(str(mp4.relative_to(output_path)))

    return patched


def patch_info_json(output_path: Path, target_size: int | None) -> list[str]:
    """视频缩放后，同步更新输出数据集 meta/info.json 中的尺寸描述。

    只处理 dtype=video 的特征，写入 shape=[3,N,N] 及 video.height/width。
    这是当前实现采用的 CHW 元数据布局，不是 _sample_to_hwc_uint8 的 HWC。
    不重新编码视频、不重算像素统计，也不改变 FPS 或 episode 索引。
    返回更新过的相机特征名；target_size 为 0/None 时不写回。"""
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    patched_cameras = []
    if target_size:
        for key, feat in info["features"].items():
            if feat.get("dtype") == "video":
                feat["shape"] = [3, target_size, target_size]
                vi = feat.get("video_info", {})
                vi["video.height"] = target_size
                vi["video.width"] = target_size
                feat["video_info"] = vi
                patched_cameras.append(key)

        if patched_cameras:
            with open(info_path, "w") as f:
                json.dump(info, f, indent=4)
            print(f"Patched camera metadata to {target_size}x{target_size}: {patched_cameras}")

    return patched_cameras


def fix_parquet_column_order(output_path: Path):
    """按 meta/info.json 的非视频 features 顺序重写数据 Parquet 列。

    视频在独立 MP4 中，因此不作为表格列加入 expected_order。
    遍历 data/<分片目录>/*.parquet；顺序一致时不写文件。
    当前实现使用 df[expected_order]：缺列会报错，多余列会被移除，
    因此不仅是显示顺序调整。写入不保留 pandas 自身的行索引，
    数据中名为 index 的正式列仍保留，用于后续逐帧奖励对齐。"""
    info_path = output_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    expected_order = [k for k, v in info["features"].items() if v["dtype"] != "video"]

    for pq_file in sorted((output_path / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq_file)
        if list(df.columns) != expected_order:
            df[expected_order].to_parquet(pq_file, index=False)


def patch_total_tasks(dataset_path: Path):
    """为旧数据集补充缺失的 total_tasks 字段，兼容 LeRobot 元数据读取。

    有 tasks.parquet 时以其行数作为任务数，没有则回退为 1。
    已有 total_tasks 时不修改、不重新校验该值。
    main 传入的是源数据目录，因此这里可能原地更新源 meta/info.json，
    而不是仅修改后续创建的奖励副本。"""
    info_path = dataset_path / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)

    if "total_tasks" not in info:
        tasks_path = dataset_path / "meta" / "tasks.parquet"
        if tasks_path.exists():
            info["total_tasks"] = len(pd.read_parquet(tasks_path))
        else:
            info["total_tasks"] = 1

        with open(info_path, "w") as f:
            json.dump(info, f, indent=4)
        print(f"Patched info.json: added total_tasks={info['total_tasks']}")


def main():
    """奖励标注主流程（在 lerobot-env 运行，不启动 Isaac Sim）。

    顺序：解析参数 → 补齐源元数据 → 构建输出副本 → 可选缩放 →
    计算逐帧奖励 → 按全局 index 写回输出 Parquet。

    三种模式：classifier 为二值成功奖励；vip 为加权视觉负距离；
    composite 为二者相加。结果用于后续 train_iql_critics.py 等程序。
    输入目录与输出目录必须独立且互不包含；当前代码不检查目录关系。
    建议使用新的输出目录，避免覆盖已有数据和奖励结果。
    """
    parser = argparse.ArgumentParser(
        description="Prepare dataset with rewards for SAC training"
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to LeRobot dataset"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output path for labeled dataset"
    )
    parser.add_argument(
        "--reward-source",
        type=str,
        default="classifier",
        choices=["classifier", "vip", "composite"],
        help="Reward source: classifier, vip, or composite (clf + vip)",
    )
    parser.add_argument(
        "--reward-model",
        type=str,
        default="outputs/reward_classifier_v2/best",
        help="Path to trained reward classifier",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classifier threshold for binary reward",
    )
    # VIP 参数只在 vip/composite 模式参与计算；目标数据不必与待标注数据相同。
    # VIP args
    parser.add_argument(
        "--vip-goal-dataset", type=str, default=None,
        help="LeRobot dataset for VIP goal images",
    )
    parser.add_argument(
        "--vip-label-dataset", type=str, default=None,
        help="Dataset with next.reward labels for VIP goal filtering",
    )
    # 启用时从标签中选 next.reward > 0.5 的成功帧；可用独立 label dataset 提供标签。
    parser.add_argument("--vip-use-labeled", action="store_true",
                        help="Use next.reward > 0.5 frames as VIP goals")
    # mean：到平均目标特征的距离；min：到多个目标特征的最近距离。
    parser.add_argument("--vip-goal-mode", type=str, default="mean",
                        choices=["mean", "min"])
    parser.add_argument("--vip-camera", type=str,
                        default="observation.images.top")
    # 不使用成功标签时，每条 episode 取末尾 N 帧作为目标；启用标签筛选后该数不生效。
    parser.add_argument("--n-goal-frames", type=int, default=5,
                        help="Final frames per episode for VIP goal")
    # 实际实现对所有 VIP 分支都乘此权重，包括纯 vip 模式，不仅是 composite。
    parser.add_argument("--w-vip", type=float, default=1.0,
                        help="VIP reward weight (for composite mode)")
    # image-size=0 保留原视频；128/256 会改写输出视频和尺寸元数据。
    # Common
    parser.add_argument(
        "--image-size",
        type=int,
        default=0,
        choices=[0, 128, 256],
        help="Resize videos to NxN (0 = no resize)",
    )
    args = parser.parse_args()

    # 按模式选择奖励计算分支；默认是 classifier，SmolVLA + IQL 的 VIP 路线需显式指定 vip。
    # Validate args
    use_clf = args.reward_source in ("classifier", "composite")
    use_vip = args.reward_source in ("vip", "composite")

    if use_vip and not args.vip_goal_dataset:
        parser.error("--vip-goal-dataset required for vip/composite reward source")

    # 注意 absolute() 只生成绝对路径，不检查源/输出是否相同或有父子关系。
    dataset_path = Path(args.dataset).absolute()
    output_path = Path(args.output).absolute()

    print(f"Loading dataset: {dataset_path}")
    print(f"Reward source: {args.reward_source}")

    # Patch missing total_tasks for older datasets
    patch_total_tasks(dataset_path)

    # 先构建带 next.reward 的输出副本；使用独立输出路径，已有输出可能被覆盖删除。
    # Step 1: Copy dataset to output with dummy rewards (zeros)
    print(f"\nCopying dataset to {output_path}...")
    dataset = LeRobotDataset(repo_id="local_dataset", root=dataset_path)

    if "next.reward" in dataset.meta.features:
        # Dataset already has next.reward (e.g. sim rewards from recording)
        print("  next.reward already exists, copying dataset as-is...")
        import shutil
        # 此分支会递归删除已有输出目录！输出不能等于源目录，也不能是源目录的祖先。
        if output_path.exists():
            shutil.rmtree(output_path)
        shutil.copytree(dataset_path, output_path)
    else:
        # 若原数据没有奖励字段，先通过 LeRobot add_features 创建带占位列的数据副本。
        dummy_rewards = np.zeros(len(dataset), dtype=np.float32)
        add_features(
            dataset=dataset,
            features={
                "next.reward": (
                    dummy_rewards,
                    {"dtype": "float32", "shape": (1,), "names": None},
                ),
            },
            output_dir=output_path,
            repo_id="local_dataset_labeled",
        )
    fix_parquet_column_order(output_path)

    # Step 2: Resize videos (before reward computation for faster decoding)
    # 只缩放输出副本；分类器读取该副本，而下面 VIP 仍读取原始分辨率的源数据。
    if args.image_size > 0:
        resize_videos(output_path, args.image_size)
        patch_info_json(output_path, args.image_size)

    # 分类器给成功信号，VIP 给视觉目标接近程度；按配置累加成逐帧奖励。
    # Step 3: Compute rewards
    total_frames = len(dataset)
    # 建立与 dataset 顺序相同的奖励数组；不同来源以加法合并，不累加旧 next.reward。
    rewards = np.zeros(total_frames, dtype=np.float32)

    if use_clf:
        # Classifier uses resized images (trained on 128x128)
        output_dataset = LeRobotDataset(repo_id="local_dataset_labeled", root=output_path)
        print(f"\nLoading reward classifier: {args.reward_model}")
        reward_model = RewardClassifier(
            args.reward_model, device="cuda", threshold=args.threshold
        )
        # 分类器贡献 0 或 1；此处权重固定为 1。
        rewards += compute_classifier_rewards(output_dataset, reward_model)

    if use_vip:
        # VIP uses original full-res images (its own Resize(224,224) handles preprocessing)
        from so101_lab.rewards.vip_reward import VIPReward
        print(f"\nLoading VIP reward (goal dataset: {args.vip_goal_dataset})...")
        # VIP 初始化包含预训练编码器和目标特征准备；模型内部负责 Resize(224,224) 等预处理。
        vip = VIPReward(
            goal_dataset_path=args.vip_goal_dataset,
            device="cuda",
            image_key=args.vip_camera,
            n_goal_frames=args.n_goal_frames,
            goal_mode=args.vip_goal_mode,
            use_labeled=args.vip_use_labeled,
            label_dataset_path=args.vip_label_dataset,
        )
        vip_rewards = compute_vip_rewards(dataset, vip, args.vip_camera)
        # 组合奖励 = 分类器二值奖励 + w_vip × VIP 奖励；数值仍可能为负，不是概率。
        rewards += vip_rewards * args.w_vip

    # 按全局 index 对齐奖励与各分片的行，避免把奖励写到错误帧。
    # Step 4: Write rewards to parquet
    print(f"\nFinal reward range: [{rewards.min():.4f}, {rewards.max():.4f}], "
          f"mean={rewards.mean():.4f}")
    print("Writing rewards to parquet...")
    for pq_file in sorted((output_path / "data").glob("*/*.parquet")):
        df = pd.read_parquet(pq_file)
        # 假定全局 index 与 dataset[i] 对齐且在 [0,total_frames)；支持按此索引写多个分片。
        indices = df["index"].tolist()
        # 覆盖副本原有 next.reward，不与旧奖励相加；这一步没有按 episode 做移位。
        df["next.reward"] = [float(rewards[i]) for i in indices]
        df.to_parquet(pq_file, index=False)

    # 当前实现只写奖励列，不在这里重新计算 meta/stats.json 的奖励统计量。
    print("Done!")


if __name__ == "__main__":
    main()
