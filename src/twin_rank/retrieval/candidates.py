"""多路召回：双塔向量召回 + 热门补充，合并去重，并统一计算召回分数。

候选总数 = n_candidates，其中 n_popular 条来自热门路，其余来自双塔。
所有候选的召回分数都用"用户向量·商品向量"重新计算，这样精排特征 recall_score 对两路一致。
Faiss 不支持逐用户排除已看商品，所以多检索 len(已看) 个再过滤。
"""
from dataclasses import dataclass

import numpy as np

from twin_rank.retrieval.popular import PopularRecall


@dataclass
class Candidates:
    item_idx: np.ndarray       # [n]
    recall_score: np.ndarray   # [n] 用户向量与物品向量的余弦相似度
    is_popular: np.ndarray     # [n] 1 = 来自热门路


class CandidateGenerator:
    def __init__(self, index, item_vecs: np.ndarray, popular: PopularRecall,
                 n_candidates: int, n_popular: int):
        self.index = index
        self.item_vecs = item_vecs
        self.popular = popular
        self.n_popular = n_popular
        self.n_tower = n_candidates - n_popular

    def generate(self, user_vecs: np.ndarray, excludes: list) -> list:
        """excludes[i]: 第 i 个用户已交互过的 item_idx 数组，不会出现在候选里。"""
        buffer = max((len(e) for e in excludes), default=0)   # 多取一些，留给过滤
        _, ids = self.index.search(user_vecs, self.n_tower + buffer)

        results = []
        for vec, row, seen in zip(user_vecs, ids, excludes):
            keep = (row > 0) & ~np.isin(row, seen)
            tower = row[keep][: self.n_tower]
            popular = self.popular.top(self.n_popular, np.concatenate([seen, tower]))
            items = np.concatenate([tower, popular]).astype("int64")
            flags = np.r_[np.zeros(len(tower)), np.ones(len(popular))].astype("int8")
            scores = (self.item_vecs[items] @ vec).astype("float32")
            results.append(Candidates(items, scores, flags))
        return results
