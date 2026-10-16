# Adapted from: facebookresearch/vip (https://github.com/facebookresearch/vip)
# Original license: CC BY-NC 4.0
# Changes: Standalone loader without vip-utils package; Resize(224,224) instead of CenterCrop

"""VIP (Value-Implicit Pre-training) reward model.

Zero-shot visual reward via pretrained ResNet50 (Ego4D).
Reward = -||phi(current_image) - phi(goal_image)||_2

Goal images: either final frames of demo episodes, or labeled success frames
(next.reward > 0.5) from a prepared dataset.

Usage:
    from so101_lab.rewards.vip_reward import VIPReward

    # Mean goal embedding (default)
    vip = VIPReward("data/recordings/figure_shape_placement_v4")

    # Min distance to closest goal
    vip = VIPReward("data/recordings/figure_shape_placement_v4", goal_mode="min")

    # Use labeled success frames instead of final frames
    vip = VIPReward("data/recordings/figure_shape_placement_v4_labeled", use_labeled=True)

    reward = vip.compute_reward(obs)  # scalar, negative (closer to 0 = better)
"""
# ============================================================================
# 【文件说明】VIP 视觉奖励模型（零样本、无需训练）
#
# 核心思想：
#   用在 Ego4D 大规模人类视频上预训练好的 ResNet50（VIP 编码器）把图像映射为
#   1024 维特征向量 phi(image)。当前画面与"目标画面"在特征空间里越近，奖励越高：
#       reward = -|| phi(当前图像) - phi(目标图像) ||_2
#   因此奖励恒为负数，越接近 0 表示越接近目标。
#
# 目标图像的两种来源：
#   1) 示范数据每条 episode 的最后 n 帧（默认，假设示范结尾即任务完成）；
#   2) 数据集中被标注为成功的帧（next.reward > 0.5，use_labeled=True）。
#
# 目标特征的两种聚合方式（goal_mode）：
#   - "mean"：把所有目标特征取平均，得到单个目标向量；
#   - "min" ：保留全部目标特征，奖励取到"最近目标"的距离。
#
# 整体流程：
#   加载编码器 -> 预先编码目标图像（并缓存到磁盘）-> 可选地计算奖励尺度
#   -> 运行时每步对当前图像编码，计算与目标的距离并返回负距离作为奖励。
# ============================================================================
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

# 预训练权重下载地址（PyTorch S3）
# Pretrained weights URL (PyTorch S3)
VIP_WEIGHTS_URL = "https://pytorch.s3.amazonaws.com/models/rl/vip/model.pt"
# 权重本地缓存目录；首次运行下载后，之后直接从这里读取
VIP_CACHE_DIR = os.path.expanduser("~/.vip/resnet50")


# 加载 Ego4D 预训练的 VIP ResNet50，输出 1024 维视觉特征。
# 首次需要下载权重至 ~/.vip/resnet50；后续从缓存读取，不在这里训练编码器。
def _load_vip_encoder(device: str = "cuda") -> nn.Module:
    """Load pretrained VIP ResNet50 encoder."""
    # 构建不带预训练权重的 ResNet50 骨架，随后手动载入 VIP 权重
    convnet = torchvision.models.resnet50(weights=None)
    # 把最后的分类全连接层替换为 2048 -> 1024 的线性层，使输出变为 1024 维嵌入
    # （需与 VIP 原始模型的结构一致，否则权重无法加载）
    convnet.fc = nn.Linear(2048, 1024)

    # 若本地没有权重文件，则创建缓存目录并从 URL 下载
    # Download weights if needed
    weights_path = os.path.join(VIP_CACHE_DIR, "model.pt")
    if not os.path.exists(weights_path):
        os.makedirs(VIP_CACHE_DIR, exist_ok=True)
        torch.hub.load_state_dict_from_url(VIP_WEIGHTS_URL, VIP_CACHE_DIR)

    # 先加载到 CPU，避免权重文件里记录的设备信息与当前机器不一致。
    # weights_only=False 表示允许反序列化任意对象，仅应对可信来源的权重文件使用。
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    # 原始权重键名带有 'module.convnet.' 前缀（来自 DataParallel 包装 + VIP 外层封装），
    # 需要去掉前缀才能与 torchvision 的 ResNet 键名对上。
    # Strip 'module.convnet.' prefix from DataParallel + VIP wrapper
    state_dict = {
        k.replace("module.convnet.", ""): v
        for k, v in ckpt["vip"].items()
    }
    convnet.load_state_dict(state_dict)
    convnet.to(device)
    # eval 模式：固定 BatchNorm 统计量，保证同一图像每次得到相同的特征
    convnet.eval()
    return convnet


class VIPReward:
    """VIP reward: embedding distance to goal image.

    Accepts numpy images (HWC uint8) from IsaacLabGymEnv.
    Returns scalar reward (negative, closer to 0 = closer to goal).
    """

    def __init__(
        self,
        goal_dataset_path: str | Path,                   # 提供目标图像的 LeRobot 数据集路径
        device: str = "cuda",                            # 推理设备
        image_key: str = "observation.images.top",       # 用哪一路相机计算嵌入
        n_goal_frames: int = 5,                          # 每条 episode 取末尾多少帧作为目标
        goal_mode: str = "mean",                         # 目标聚合方式："mean" 或 "min"
        use_labeled: bool = False,                       # 是否改用带成功标签的帧作为目标
        label_dataset_path: str | Path | None = None,    # 提供成功标签的数据集（默认与目标数据集相同）
        normalize: bool = False,                         # 是否把奖励归一化到约 [-1, 0]
        scale_dataset_path: str | Path | None = None,    # 用于估计归一化尺度的数据集
        reward_clip: tuple[float, float] = (-2.0, 0.0),  # 归一化后的裁剪区间（下界, 上界）
    ):
        """
        Args:
            goal_dataset_path: Path to LeRobot dataset (source of images).
            device: Torch device.
            image_key: Which camera to use for embeddings.
            n_goal_frames: Final frames per episode for goal (ignored if use_labeled=True).
            goal_mode: "mean" — single averaged goal embedding,
                       "min" — keep all goal embeddings, reward = min distance.
            use_labeled: If True, use frames with next.reward > 0.5 as goals.
            label_dataset_path: Dataset with next.reward labels (default: same as goal_dataset_path).
                                Images are always taken from goal_dataset_path.
            normalize: If True, normalize rewards to [-1, 0] using scale from data.
            scale_dataset_path: Dataset with precomputed VIP rewards for normalization
                                scale (typically --demo-dataset). Falls back to sampling
                                from goal_dataset if None.
            reward_clip: Clip range after normalization (default: (-2.0, 0.0)).
        """
        self.device = device
        self.image_key = image_key
        self.goal_mode = goal_mode
        self.normalize = normalize
        self.reward_clip = reward_clip
        self.reward_scale = 1.0   # 默认尺度为 1（即不缩放）；若 normalize=True 会在下方更新
        # updated below if normalize=True

        # 图像统一缩放到 224×224，再按 ImageNet 均值/方差归一化；此处采用 Resize 而非中心裁剪。
        # 说明：Resize 会保留完整视野（但可能改变长宽比），CenterCrop 会裁掉边缘，
        # 对于机械臂场景，边缘可能包含重要物体，所以这里选择 Resize。
        # 处理顺序：Resize -> 转 float32（uint8 自动除以 255 到 [0,1]）-> 标准化。
        # Preprocessing: resize to 224x224 + ImageNet normalize
        self.preprocess = T.Compose([
            T.Resize((224, 224)),
            T.ConvertImageDtype(torch.float32),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # 加载 VIP 编码器（冻结，仅用于前向特征提取）
        # Load encoder
        self.encoder = _load_vip_encoder(device)

        # 成功目标可以来自同一数据集或独立目标数据；预先编码以避免每一步重复处理目标图像。
        # 结果带磁盘缓存：第二次初始化时可直接读取，显著缩短启动时间。
        # Precompute goal embeddings from dataset (with disk cache)
        goal_embeddings = self._load_or_compute_goal_embeddings(
            goal_dataset_path, use_labeled, label_dataset_path, n_goal_frames
        )
        # 根据 goal_mode 决定如何保存目标特征：
        #   mean：只保存一个平均向量，计算最快；
        #   min ：保留全部 N 个向量，奖励取最近的一个，更适合目标有多种成功形态的情况。
        if goal_mode == "mean":
            self.goal_embedding = goal_embeddings.mean(dim=0)  # (1024,)
            self.goal_embeddings = None
            norm = torch.norm(self.goal_embedding).item()
            print(f"  Goal mode: mean (single embedding, norm={norm:.2f})")
        else:
            # 注意：任何不等于 "mean" 的取值都会走 min 分支（代码没有对非法值做校验）
            self.goal_embedding = None
            self.goal_embeddings = goal_embeddings  # (N, 1024)
            print(f"  Goal mode: min (closest of {len(goal_embeddings)} embeddings)")

        # 若启用归一化，则根据数据估计一个固定的奖励尺度
        # Compute normalization scale
        if normalize:
            self.reward_scale = self._compute_reward_scale(
                scale_dataset_path, goal_dataset_path
            )
            print(f"  Normalize: scale={self.reward_scale:.2f}, clip={reward_clip}")

        print(f"VIP reward initialized (camera={image_key})")

    def _load_or_compute_goal_embeddings(
        self,
        goal_dataset_path: str | Path,
        use_labeled: bool,
        label_dataset_path: str | Path | None,
        n_goal_frames: int,
    ) -> torch.Tensor:
        """Load cached goal embeddings or compute and cache them.

        Cache file: <goal_dataset_path>/vip_goal_cache_<key>.pt
        Key encodes: image_key, use_labeled, label_path, n_goal_frames.
        """
        import hashlib

        lbl_path = str(label_dataset_path or goal_dataset_path)
        # 缓存键仅包含相机键、标签选项、标签路径和末帧数量，不包含数据内容哈希。
        # 若原目录图像或成功标签改变，需要重新生成对应缓存，不能认为会自动失效。
        cache_key_str = f"{self.image_key}|labeled={use_labeled}|lbl={lbl_path}|n={n_goal_frames}"
        # 对配置字符串做 MD5 并取前 8 位，作为缓存文件名的一部分，不同配置对应不同缓存
        cache_hash = hashlib.md5(cache_key_str.encode()).hexdigest()[:8]
        cache_path = Path(goal_dataset_path) / f"vip_goal_cache_{cache_hash}.pt"

        # 命中缓存则直接加载并返回。weights_only=True 只允许加载纯张量，更安全。
        if cache_path.exists():
            goal_embeddings = torch.load(cache_path, map_location=self.device, weights_only=True)
            print(f"  Loaded cached goal embeddings: {cache_path} ({goal_embeddings.shape[0]} embeddings)")
            return goal_embeddings
        
        # 未命中缓存：从头计算
        # Compute from scratch
        if use_labeled:
            # 方式一：用 next.reward > 0.5 的成功帧作为目标
            goal_embeddings = self._compute_goal_from_labels(
                goal_dataset_path, label_dataset_path or goal_dataset_path
            )
        else:
            # 方式二：用每条 episode 的末尾若干帧作为目标
            goal_embeddings = self._compute_goal_from_finals(
                goal_dataset_path, n_goal_frames
            )
            
        # 保存缓存（先转到 CPU，保证缓存文件与设备无关）
        # Save cache
        torch.save(goal_embeddings.cpu(), cache_path)
        print(f"  Cached goal embeddings: {cache_path}")

        return goal_embeddings.to(self.device)

    def _encode_frame(self, dataset, frame_idx: int) -> torch.Tensor:
        """Encode a single dataset frame → (1, 1024) embedding."""
        frame = dataset[frame_idx]
        img = frame[self.image_key]  # (C, H, W) float [0, 1] from LeRobot
        img_uint8 = (img * 255).to(torch.uint8)
        img_preprocessed = self.preprocess(img_uint8).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return self.encoder(img_preprocessed)  # (1, 1024)

    def _compute_goal_from_labels(
        self, image_dataset_path: str | Path, label_dataset_path: str | Path
    ) -> torch.Tensor:
        """Extract goal embeddings using labels from one dataset, images from another.

        Args:
            image_dataset_path: Dataset with full-res images for encoding.
            label_dataset_path: Dataset with next.reward column for filtering.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        label_ds = LeRobotDataset(repo_id="local", root=str(label_dataset_path))
        if str(image_dataset_path) == str(label_dataset_path):
            image_ds = label_ds
        else:
            image_ds = LeRobotDataset(repo_id="local", root=str(image_dataset_path))

        # 标签只用于选帧，图像仍来自 image_dataset；两个数据集的全局帧索引必须对应。
        # Find success frame indices from label dataset
        success_indices = []
        for i in range(len(label_ds)):
            r = float(label_ds.hf_dataset[i]["next.reward"])
            if r > 0.5:
                success_indices.append(i)

        if not success_indices:
            raise ValueError(f"No success frames (next.reward > 0.5) in {label_dataset_path}")

        print(f"  Labels from: {label_dataset_path}")
        print(f"  Images from: {image_dataset_path}")
        print(f"  Found {len(success_indices)} success frames")

        # 成功帧过多时最多编码 1000 帧，以限制初始化时间；这是帧数，不是成功 episode 数。
        # Subsample if too many (encode max 1000)
        if len(success_indices) > 1000:
            step = len(success_indices) // 1000
            success_indices = success_indices[::step][:1000]
            print(f"  Subsampled to {len(success_indices)} frames")

        # Encode images from image_ds (not label_ds)
        embeddings = []
        for idx in success_indices:
            emb = self._encode_frame(image_ds, idx)
            embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # (N, 1024)

    # 没有成功标签时，从每条 episode 尾部选目标帧。
    # 这隐含末尾画面接近目标的假设；若失败示范也以该方式取目标，奖励含义可能失真。
    def _compute_goal_from_finals(
        self, dataset_path: str | Path, n_goal_frames: int
    ) -> torch.Tensor:
        """Extract goal embeddings from final frames of each episode."""
        import pandas as pd
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local", root=str(dataset_path))

        ep_parquet = Path(dataset_path) / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
        ep_df = pd.read_parquet(ep_parquet)
        n_episodes = len(ep_df)
        print(f"  Extracting goal from {n_episodes} episodes "
              f"({n_goal_frames} final frames each)...")

        embeddings = []
        for _, row in ep_df.iterrows():
            ep_start = int(row["dataset_from_index"])
            ep_end = int(row["dataset_to_index"])
            ep_len = ep_end - ep_start

            n_take = min(n_goal_frames, ep_len)
            for offset in range(n_take):
                frame_idx = ep_end - 1 - offset
                emb = self._encode_frame(dataset, frame_idx)
                embeddings.append(emb)

        return torch.cat(embeddings, dim=0)  # (N, 1024)

    # VIP 自身的固定奖励尺度：优先使用预标注奖励的负 5% 分位数，
    # 否则对目标数据抽样图像距离取 95% 分位数，并设置最小尺度 1。
    # 它不是 PPO 服务端 RunningMeanStd 的在线奖励归一化。
    def _compute_reward_scale(
        self,
        scale_dataset_path: str | Path | None,
        goal_dataset_path: str | Path,
    ) -> float:
        """Compute reward normalization scale.

        If scale_dataset has next.reward (precomputed VIP rewards),
        use 5th percentile. Otherwise sample random frames from goal
        dataset and measure distances to goal embedding.
        """
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        # Try scale dataset with precomputed VIP rewards (typically --demo-dataset)
        if scale_dataset_path is not None:
            scale_ds = LeRobotDataset(repo_id="local", root=str(scale_dataset_path))
            if "next.reward" in scale_ds.meta.features:
                rewards = [
                    float(scale_ds.hf_dataset[i]["next.reward"])
                    for i in range(len(scale_ds))
                ]
                scale = -float(np.percentile(rewards, 5))
                print(f"  Scale from {scale_dataset_path}: p5={-scale:.2f}")
                return max(scale, 1.0)  # safety floor

        # Fallback: sample random frames from goal dataset
        print("  No precomputed rewards, sampling 100 random frames...")
        image_ds = LeRobotDataset(repo_id="local", root=str(goal_dataset_path))
        rng = np.random.default_rng(42)
        n_samples = min(100, len(image_ds))
        indices = rng.choice(len(image_ds), size=n_samples, replace=False)

        dists = []
        for idx in indices:
            emb = self._encode_frame(image_ds, int(idx)).squeeze(0)
            # mean 比较当前特征与目标特征均值；min 比较到多个目标的最小距离。
            # 两者几何含义不同：先平均特征再算距离不等于对多个距离求平均。
            if self.goal_mode == "mean":
                d = torch.norm(emb - self.goal_embedding).item()
            else:
                d = torch.norm(self.goal_embeddings - emb.unsqueeze(0), dim=1).min().item()
            dists.append(d)

        scale = float(np.percentile(dists, 95))
        print(f"  Scale from sampled frames: p95={scale:.2f}")
        return max(scale, 1.0)

    # 取指定相机的 HWC 图像，变为 CHW 并增加 batch 维；统一预处理后送入编码器。
    def _prepare_image(self, obs: dict) -> torch.Tensor:
        """Convert raw observation image to preprocessed tensor."""
        img = obs[self.image_key]  # (H, W, 3) uint8 numpy

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img)

        # (H, W, C) -> (C, H, W)
        if img.dim() == 3 and img.shape[-1] == 3:
            img = img.permute(2, 0, 1)

        # Preprocess: resize 224x224, /255, ImageNet normalize
        img = self.preprocess(img.unsqueeze(0).to(self.device))
        return img  # (1, 3, 224, 224)

    @torch.no_grad()
    # 奖励为视觉欧氏距离的负值，不是成功概率，也不是前后两帧距离之差。
    # 靠近目标通常使奖励趋近 0，但相似画面不保证物体真正入槽或夹爪已经松开。
    def compute_reward(self, obs: dict) -> float:
        """Compute VIP reward: -||phi(current) - phi(goal)||_2.

        Args:
            obs: Dict with image arrays, e.g.:
                {"observation.images.top": (H, W, 3) uint8 numpy}

        Returns:
            Scalar reward (negative, closer to 0 = better).
            If normalize=True, scaled to [-1, 0] and clipped.
        """
        img = self._prepare_image(obs)
        emb = self.encoder(img).squeeze(0)  # (1024,)

        # mean 比较当前特征与目标特征均值；min 比较到多个目标的最小距离。
        # 两者几何含义不同：先平均特征再算距离不等于对多个距离求平均。
        if self.goal_mode == "mean":
            raw = -torch.norm(emb - self.goal_embedding).item()
        else:
            dists = torch.norm(self.goal_embeddings - emb.unsqueeze(0), dim=1)  # (N,)
            raw = -dists.min().item()

        # 仅启用 VIP 自身 normalize 时才除以固定尺度并裁剪；默认可返回较大的负数。
        if self.normalize:
            normalized = raw / self.reward_scale
            return max(self.reward_clip[0], min(normalized, self.reward_clip[1]))
        return raw
