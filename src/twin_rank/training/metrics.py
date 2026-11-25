"""排序指标：Recall@K 与 NDCG@K（每个用户只有 1 个真实商品）。

指标名用 recall_at_50 而不是 recall@50，因为 MLflow 的指标名不允许 '@'。
"""

import torch


def ranking_metrics(topk: torch.Tensor, target: torch.Tensor, ks: list) -> dict:
    """
    topk: [B, Kmax] 召回的物品 idx；target: [B] 真实物品 idx（每人 1 个）。
    返回本 batch 的 recall@K / ndcg@K 之和（由调用方除以样本总数）。
    指标名不含 '@'，因为 MLflow 不允许。
    """
    hits = topk == target.unsqueeze(1)
    positions = torch.arange(topk.shape[1], device=topk.device)
    discount = 1.0 / torch.log2(positions + 2.0)
    out = {}
    for k in ks:
        hit_k = hits[:, :k]
        out[f"recall_at_{k}"] = hit_k.any(dim=1).float().sum().item()
        out[f"ndcg_at_{k}"] = (hit_k.float() * discount[:k]).sum().item()
    return out
