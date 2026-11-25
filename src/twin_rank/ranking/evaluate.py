"""对比三种方案的端到端指标：仅召回 / 召回+精排 / 召回+精排+重排，并给出候选命中率（精排效果的上限）。
"""
import numpy as np
import pandas as pd

from twin_rank.ranking.metrics import metrics_from_positions, positions_by_score


def _positions_after_rerank(frame: pd.DataFrame, scores: np.ndarray, reranker, top_k: int,
                            n_users: int) -> np.ndarray:
    user_row = frame["user_row"].to_numpy()
    items, labels = frame["item_idx"].to_numpy(), frame["label"].to_numpy()
    cuts = np.flatnonzero(np.diff(user_row)) + 1
    starts, ends = np.r_[0, cuts], np.r_[cuts, len(frame)]

    positions = np.full(n_users, np.inf)
    for s, e in zip(starts, ends):
        target = items[s:e][labels[s:e] == 1]
        if len(target) == 0:
            continue
        final = reranker.rerank(items[s:e], scores[s:e], top_k)
        hit = np.flatnonzero(final == target[0])
        if len(hit):
            positions[user_row[s]] = hit[0]
    return positions


def evaluate_variants(frame: pd.DataFrame, rank_scores: np.ndarray, reranker,
                      n_users: int, ks: list) -> dict:
    user_row, label = frame["user_row"].to_numpy(), frame["label"].to_numpy()
    recall_pos = positions_by_score(user_row, label, frame["recall_score"].to_numpy(), n_users)
    rank_pos = positions_by_score(user_row, label, rank_scores, n_users)
    full_pos = _positions_after_rerank(frame, rank_scores, reranker, max(ks), n_users)

    results = {"candidate_hit_rate": float(np.isfinite(recall_pos).mean())}
    for name, positions in (("recall_only", recall_pos), ("ranker", rank_pos), ("full", full_pos)):
        for metric, value in metrics_from_positions(positions, ks).items():
            results[f"{name}_{metric}"] = value
    return results
