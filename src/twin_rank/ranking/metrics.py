"""端到端指标（numpy 版）：先求目标商品在排序结果中的位置，再算 Recall/NDCG。目标不在候选里则位置为 inf（算未命中）。
"""
import numpy as np


def positions_by_score(user_row: np.ndarray, label: np.ndarray, score: np.ndarray,
                       n_users: int) -> np.ndarray:
    """按分数降序后，每个用户目标物品的位置（0 起）；目标不在候选里则为 inf。"""
    order = np.lexsort((-score, user_row))
    rows, labels = user_row[order], label[order]
    starts = np.searchsorted(rows, np.arange(n_users))
    hit = np.flatnonzero(labels == 1)
    positions = np.full(n_users, np.inf)
    positions[rows[hit]] = hit - starts[rows[hit]]
    return positions


def metrics_from_positions(positions: np.ndarray, ks: list) -> dict:
    out = {}
    for k in ks:
        in_top = positions < k
        out[f"recall_at_{k}"] = float(in_top.mean())
        out[f"ndcg_at_{k}"] = float(np.where(in_top, 1.0 / np.log2(positions + 2.0), 0.0).mean())
    return out
