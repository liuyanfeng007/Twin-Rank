"""用户行为序列存储（进程内）。生产环境可换成 Redis / 特征平台。
"""
import numpy as np
import pandas as pd


class HistoryStore:
    def __init__(self, encoded: pd.DataFrame):
        users = encoded["user_idx"].to_numpy()                   # 已按 (user, time) 排序
        items = encoded["item_idx"].to_numpy()
        cuts = np.flatnonzero(np.diff(users)) + 1
        self.seqs = dict(zip(users[np.r_[0, cuts]], np.split(items, cuts)))

    def get(self, user_idx: int) -> np.ndarray:
        return self.seqs.get(user_idx, np.zeros(0, dtype="int64"))

    def append(self, user_idx: int, item_idx: int) -> None:
        self.seqs[user_idx] = np.append(self.get(user_idx), item_idx)

    def users_with_min_len(self, min_len: int) -> list:
        return [u for u, seq in self.seqs.items() if len(seq) >= min_len]
