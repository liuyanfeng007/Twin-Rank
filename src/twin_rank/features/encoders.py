"""ID 编码器：原始字符串 ID <-> 连续整数。

编号 0 预留给 padding / 未知（embedding 表的 padding_idx=0），有效 ID 从 1 开始。
编码器只用训练集拟合；验证/测试里没见过的用户或商品会得到 0，随后被剔除（冷启动不在离线评估范围内）。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd


class IdEncoder:
    """原始 ID -> 连续整数。0 预留给 padding / 未知，所以有效 ID 从 1 开始。"""

    def __init__(self, mapping: dict = None):
        self.mapping = mapping or {}

    def fit(self, values) -> "IdEncoder":
        uniques = pd.unique(pd.Series(values))
        self.mapping = {v: i + 1 for i, v in enumerate(uniques)}
        return self

    def transform(self, values) -> np.ndarray:
        return pd.Series(values).map(self.mapping).fillna(0).astype("int64").to_numpy()

    def __len__(self) -> int:
        """embedding 表大小（含 padding 位）。"""
        return len(self.mapping) + 1

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.mapping, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "IdEncoder":
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))
