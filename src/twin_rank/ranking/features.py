"""精排特征。离线造样本与在线打分共用这一份代码，避免训练/服务特征不一致（training-serving skew）。

特征分四类: 召回信息(分数/名次/是否热门路)、物品侧(类目/价格桶/热度)、用户侧(历史长度)、
用户-物品交叉(历史里同类目/同品牌的个数、与最近一次是否同品牌、价格与偏好的偏差)。
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from twin_rank.retrieval.candidates import Candidates

FEATURE_NAMES = [
    "recall_score", "recall_rank", "is_popular",            # 召回阶段信息
    "cat_idx", "price_bucket", "pop_bucket", "item_log_pop",  # 物品侧
    "user_hist_len",                                        # 用户侧
    "cat_match_cnt", "store_match_cnt", "last_store_match", "price_diff",  # 用户-物品交叉
]
CATEGORICAL = ["cat_idx"]


@dataclass
class ItemTables:
    item_feats: np.ndarray     # [n_items, 4] = cat, store, price_bucket, pop_bucket
    item_log_pop: np.ndarray   # [n_items] log(1 + 训练集点击数)


def _rank_within_user(score: np.ndarray, user_row: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """每个用户内按分数降序的名次（从 1 开始）。user_row 已按用户连续排列，比 pandas groupby 快得多。"""
    order = np.lexsort((-score, user_row))            # 先按用户、再按分数降序（稳定）
    starts = np.repeat(np.cumsum(sizes) - sizes, sizes)
    ranks = np.empty(len(score), dtype="int64")
    ranks[order] = np.arange(len(score)) - starts + 1
    return ranks


def build_ranking_features(hist: np.ndarray, cands: list, tables: ItemTables) -> pd.DataFrame:
    """
    hist: [B, L] 每个用户的历史；cands: 长度为 B 的候选列表。
    返回每个 (用户, 候选) 一行；user_row 是用户在本批中的序号。
    """
    sizes = np.array([len(c.item_idx) for c in cands])
    user_row = np.repeat(np.arange(len(cands)), sizes)
    item = np.concatenate([c.item_idx for c in cands])
    score = np.concatenate([c.recall_score for c in cands])
    is_popular = np.concatenate([c.is_popular for c in cands])
    recall_rank = _rank_within_user(score, user_row, sizes)

    feats = tables.item_feats
    cand = feats[item]                                   # [M, 4]
    h = hist[user_row]                                   # [M, L]
    valid = h > 0
    hist_len = valid.sum(axis=1)

    cat_match = ((feats[h, 0] == cand[:, 0:1]) & valid).sum(axis=1)
    store_match = ((feats[h, 1] == cand[:, 1:2]) & valid).sum(axis=1)
    mean_price = (feats[h, 2] * valid).sum(axis=1) / np.maximum(hist_len, 1)

    last = hist[:, -1][user_row]                         # 历史左侧补 0，最后一列是最近一次
    last_store = np.where(last > 0, feats[last, 1], -1)

    return pd.DataFrame({
        "user_row": user_row,
        "item_idx": item,
        "recall_score": score,
        "recall_rank": recall_rank,
        "is_popular": is_popular,
        "cat_idx": cand[:, 0],
        "price_bucket": cand[:, 2],
        "pop_bucket": cand[:, 3],
        "item_log_pop": tables.item_log_pop[item],
        "user_hist_len": hist_len,
        "cat_match_cnt": cat_match,
        "store_match_cnt": store_match,
        "last_store_match": (last_store == cand[:, 1]).astype("int8"),
        "price_diff": np.abs(cand[:, 2] - mean_price),
    })
