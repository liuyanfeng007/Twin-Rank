"""晋升决策（纯逻辑，不依赖 MLflow，方便测试）。

规则：新版本只有在"与 champion 评估口径相同"且"目标指标超过 champion + min_improvement"时才晋升。
口径不同、champion 没有指标时，既不能说更好也不能说更差，所以不自动晋升，需要人工判断。
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class Decision:
    promote: bool
    comparable: bool      # 能否与当前 champion 比较
    reason: str


def decide(metric: str, new_metrics: dict, new_fingerprint: dict,
           champion_metrics: Optional[dict], champion_fingerprint: Optional[dict],
           min_improvement: float = 0.0) -> Decision:
    if metric not in new_metrics:
        raise ValueError(f"registry.promote_metric={metric!r} 不在可用指标里: {sorted(new_metrics)}")
    if champion_metrics is None:
        return Decision(True, True, "目前没有 champion，首个版本直接晋升")
    if metric not in champion_metrics:
        return Decision(False, False, f"当前 champion 没有 {metric} 指标（可能是旧流程注册的），无法比较")
    if new_fingerprint != champion_fingerprint:
        return Decision(False, False, "数据或评估设置与当前 champion 不一致（数据集、测试用户数、K 值或随机种子不同），指标不可比")

    new, old = new_metrics[metric], champion_metrics[metric]
    if new > old + min_improvement:
        return Decision(True, True, f"{metric}: {old:.4f} -> {new:.4f}，超过 champion")
    return Decision(False, True, f"{metric}: 新版本 {new:.4f} 未超过 champion {old:.4f}"
                                 f"（要求至少高出 {min_improvement}）")
