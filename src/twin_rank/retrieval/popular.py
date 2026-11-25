"""热门召回：按训练集点击次数排序，用于补充召回路与新用户冷启动。
"""

import numpy as np


class PopularRecall:
    """热门召回：按训练集点击次数排序。用于补充召回路与冷启动用户。"""

    def __init__(self, train_item_idx: np.ndarray, n_items: int):
        counts = np.bincount(train_item_idx, minlength=n_items)
        counts[0] = 0                                   # padding
        order = np.argsort(-counts, kind="stable")
        self.order = order[counts[order] > 0]

    def top(self, n: int, exclude=()) -> np.ndarray:
        exclude = np.asarray(exclude, dtype="int64")
        head = self.order[: n + len(exclude)]           # 最多被剔除 len(exclude) 个，这里足够
        return head[~np.isin(head, exclude)][:n]
