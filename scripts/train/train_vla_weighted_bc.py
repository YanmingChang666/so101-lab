"""Weighted BC training for SmolVLA using IQL advantage weights.

Monkey-patches SmolVLAPolicy.forward() to multiply per-sample losses by
`iql_weight` from the dataset before aggregation. Everything else is
standard lerobot-train.

Requires `iql_weight` column in dataset parquet (see compute_iql_weights.py).

Usage:
    eval "$(./activate_lerobot.sh)"
    python scripts/train/train_vla_weighted_bc.py \
        --policy.type=smolvla \
        --dataset.repo_id=data/recordings/figure_shape_placement_v5_vip \
        ...

    All lerobot-train arguments are supported.
"""

# 中文流程说明（完整顺序见 docs/workflow/training_pipeline_cn.md）
# 定位：SmolVLA 离线优化第四步：用 IQL 优势权重微调策略。
# 运行环境与输入：lerobot-env；传入已有 SmolVLA 权重、带 iql_weight 的数据和 LeRobot 训练参数。
# 输出与下游：输出 LeRobot 格式微调检查点；再用 eval_vla_policy.py 或 sweep_vla_eval.py 评估。
# 关键逻辑与边界：通过运行时替换 SmolVLAPolicy.forward，在逐样本损失聚合前乘权重。
#     没有 iql_weight 时会退回普通 loss；不会自动给 ACT 添加 IQL 功能。
#     依赖 LeRobot 暴露 losses_after_rm_padding，升级后需核对这个接口。

import logging

import torch
from torch import Tensor

_original_forward = None


def _weighted_forward(self, batch: dict[str, Tensor], noise=None, time=None):
    """SmolVLAPolicy.forward with IQL weight support."""
    loss, loss_dict = _original_forward(self, batch, noise, time)

    # 缺少权重时直接返回普通 BC loss，不会报错；实验时应确认数据确实携带该列。
    if "iql_weight" not in batch:
        return loss, loss_dict

    # Recompute loss with per-sample weights.
    # losses shape after forward: (B, chunk_size, action_dim)
    losses = loss_dict["losses_after_rm_padding"]
    weights = batch["iql_weight"].to(losses.device).float()  # (B,)
    # 把逐样本权重扩展为 (B,1,1)，对同一样本的所有时间步和动作维度使用同一权重。
    weights = weights[:, None, None]  # (B, 1, 1) for broadcasting

    # 先加权再求均值；没有除以权重之和，因此权重尺度也会改变整体梯度尺度。
    weighted_loss = (losses * weights).mean()

    loss_dict["loss_unweighted"] = loss.item()
    loss_dict["loss"] = weighted_loss.item()
    loss_dict["iql_weight_mean"] = weights.mean().item()

    return weighted_loss, loss_dict


def patch_smolvla():
    """Monkey-patch SmolVLAPolicy.forward for weighted BC."""
    global _original_forward
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    _original_forward = SmolVLAPolicy.forward
    SmolVLAPolicy.forward = _weighted_forward
    logging.info("Patched SmolVLAPolicy.forward for weighted BC (iql_weight)")


def main():
    from lerobot.scripts.lerobot_train import train
    from lerobot.utils.import_utils import register_third_party_plugins

    register_third_party_plugins()
    patch_smolvla()
    train()


if __name__ == "__main__":
    main()
