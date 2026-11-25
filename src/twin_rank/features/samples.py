"""把交互序列转成训练样本 (用户, 历史序列, 目标商品)。

对用户的第 j 次交互，历史 = 该用户在它之前的最近 max_len 次交互（左侧补 0）。
因此验证样本的历史只含训练数据，测试样本的历史含训练+验证数据，符合时间顺序。
"""
import numpy as np
import pandas as pd


def build_history(user_idx: np.ndarray, item_idx: np.ndarray, max_len: int) -> np.ndarray:
    """
    要求输入已按 (user_idx, timestamp) 排序。
    第 j 行的历史 = 同一用户在它之前的最多 max_len 个物品，左侧用 0 补齐。
    """
    n = len(item_idx)
    position = pd.Series(user_idx).groupby(user_idx).cumcount().to_numpy()
    offsets = np.arange(-max_len, 0)
    source = np.arange(n)[:, None] + offsets[None, :]
    valid = (position[:, None] + offsets[None, :]) >= 0   # 未越过该用户的第一条
    hist = np.where(valid, item_idx[np.clip(source, 0, n - 1)], 0)
    return hist.astype(np.int32)


def split_arrays(df: pd.DataFrame, hist: np.ndarray, split: str) -> dict:
    mask = (df["split"] == split).to_numpy()
    return {
        "user_idx": df["user_idx"].to_numpy()[mask],
        "item_idx": df["item_idx"].to_numpy()[mask],
        "hist": hist[mask],
    }


def pad_history(seq: np.ndarray, max_len: int) -> np.ndarray:
    """在线服务用：取最近 max_len 个物品，左侧补 0，与离线样本格式一致。"""
    tail = seq[-max_len:]
    out = np.zeros(max_len, dtype=np.int32)
    out[max_len - len(tail):] = tail
    return out
