"""深度精排的稠密特征：取自精排特征表中的数值列，做 log 变换后标准化。

类别型字段（商品 ID、类目、品牌、价格桶、热度桶）不在这里：网络内部用 item_idx 查
物品特征表，再各自过 embedding，所以训练和线上都只需传 item_idx。
"""
import numpy as np
import pandas as pd

DENSE_FEATURES = [
    "recall_score", "recall_rank", "is_popular", "item_log_pop", "user_hist_len",
    "cat_match_cnt", "store_match_cnt", "last_store_match", "price_diff",
]
LOG_FEATURES = {"recall_rank", "user_hist_len", "cat_match_cnt", "store_match_cnt"}   # 长尾计数 -> log1p


def raw_dense(frame: pd.DataFrame) -> np.ndarray:
    columns = []
    for name in DENSE_FEATURES:
        values = frame[name].to_numpy().astype("float32")
        columns.append(np.log1p(values) if name in LOG_FEATURES else values)
    return np.stack(columns, axis=1)


class Scaler:
    """标准化：(x - 均值) / 标准差，均值和标准差只用训练集计算。标准化后的 0 就是"训练集均值"。"""

    def __init__(self, mean: np.ndarray, std: np.ndarray):
        self.mean, self.std = mean.astype("float32"), std.astype("float32")

    @classmethod
    def fit(cls, x: np.ndarray) -> "Scaler":
        return cls(x.mean(axis=0), x.std(axis=0) + 1e-6)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.std).astype("float32")

    def state(self) -> dict:                      # 存成 list，torch.load(weights_only) 才能读
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_state(cls, state: dict) -> "Scaler":
        return cls(np.array(state["mean"]), np.array(state["std"]))
