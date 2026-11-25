"""用召回结果构造精排样本：候选里是真实目标 = 1，其余 = 0，同一用户的行连续排列（LambdaRank 需要分组）。

注意: 精排样本来自"验证集用户"，因为双塔没在这些目标上训练过，候选分布更接近线上。
"""
import logging
import time

import numpy as np
import pandas as pd

from twin_rank.ranking.features import build_ranking_features

logger = logging.getLogger(__name__)


def seen_items(seen_csr, user_idx: int) -> np.ndarray:
    return seen_csr.indices[seen_csr.indptr[user_idx]:seen_csr.indptr[user_idx + 1]].astype("int64")


def take_users(arrays: dict, rows: np.ndarray) -> dict:
    return {key: value[rows] for key, value in arrays.items()}


def build_ranker_frame(arrays: dict, seen_csr, stack, chunk_size: int = 2000) -> pd.DataFrame:
    """分块处理，控制内存。user_row 为用户在 arrays 中的全局行号，且同一用户的行连续。"""
    frames = []
    n = len(arrays["user_idx"])
    for start in range(0, n, chunk_size):
        sl = slice(start, start + chunk_size)
        users, hist, target = arrays["user_idx"][sl], arrays["hist"][sl], arrays["item_idx"][sl]

        t0 = time.perf_counter()
        vecs = stack.user_encoder.encode(users, hist)
        t1 = time.perf_counter()
        cands = stack.generator.generate(vecs, [seen_items(seen_csr, u) for u in users])
        t2 = time.perf_counter()
        frame = build_ranking_features(hist, cands, stack.tables)
        t3 = time.perf_counter()
        logger.info("chunk %d/%d | 用户向量 %.1fs | 召回 %.1fs | 特征 %.1fs",
                    start // chunk_size + 1, -(-n // chunk_size), t1 - t0, t2 - t1, t3 - t2)

        frame["label"] = (frame["item_idx"].to_numpy() == target[frame["user_row"].to_numpy()]).astype("int8")
        frame["user_row"] += start
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def drop_groups_without_positive(frame: pd.DataFrame) -> pd.DataFrame:
    """
    只保留候选里包含真实目标的用户。LambdaRank 对没有正样本的分组学不到排序信号，
    而且 LightGBM 会把这类分组的 NDCG 记为 1，导致验证指标虚高。仅用于训练/验证，评估时不要用。
    """
    has_positive = frame.groupby("user_row")["label"].transform("max") == 1
    return frame[has_positive].reset_index(drop=True)
