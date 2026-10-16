"""按需加载奖励模型，VIP/PPO 不需要安装 SAC 分类器依赖。"""

from importlib import import_module

__all__ = [
    "RewardClassifier",
    "VIPReward",
]


def __getattr__(name):
    modules = {"RewardClassifier": ".classifier", "VIPReward": ".vip_reward"}
    if name not in modules:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value
