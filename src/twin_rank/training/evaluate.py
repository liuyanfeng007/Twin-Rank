"""离线评估：对全量物品做精确检索（暴力内积），计算 Recall@K / NDCG@K。

做法: 用户向量 @ 全量物品向量 -> 屏蔽 padding 与已交互商品 -> 取 Top-K -> 与真实商品比对。
"""
import numpy as np
import torch

from twin_rank.training.metrics import ranking_metrics


def _seen_positions(seen, user_ids: np.ndarray, device):
    coo = seen[user_ids].tocoo()
    return torch.from_numpy(coo.row).long().to(device), torch.from_numpy(coo.col).long().to(device)


@torch.no_grad()
def evaluate(model, arrays: dict, seen, ks: list, device, batch_size: int = 2048) -> dict:
    model.eval()
    item_vecs = model.all_item_vectors().to(device)
    n = len(arrays["user_idx"])
    k_max = max(ks)
    totals = {}

    for start in range(0, n, batch_size):
        sl = slice(start, start + batch_size)
        users_np = arrays["user_idx"][sl]
        users = torch.from_numpy(users_np).long().to(device)
        hist = torch.from_numpy(arrays["hist"][sl]).long().to(device)
        target = torch.from_numpy(arrays["item_idx"][sl]).long().to(device)

        scores = model.user_vector(users, hist) @ item_vecs.T
        scores[:, 0] = float("-inf")                       # padding 位不参与
        rows, cols = _seen_positions(seen, users_np, device)
        scores[rows, cols] = float("-inf")                 # 已交互物品不参与

        topk = scores.topk(k_max, dim=1).indices
        for name, value in ranking_metrics(topk, target, ks).items():
            totals[name] = totals.get(name, 0.0) + value

    return {name: value / n for name, value in totals.items()}
