"""Faiss 向量索引。

* 索引里不放 padding 行，所以 Faiss 行号 + 1 = item_idx；
* flat = 精确内积检索；hnsw = 近似检索（物品很多时更快）；
* 向量已 L2 归一化，内积即余弦相似度。
"""
import torch  # noqa: F401  必须先于 faiss 加载，避免 macOS 双 libomp 冲突
import faiss
import numpy as np


class VectorIndex:
    def __init__(self, index):
        self.index = index

    @classmethod
    def build(cls, item_vecs: np.ndarray, index_type: str = "flat",
              hnsw_m: int = 32, ef_search: int = 128) -> "VectorIndex":
        vecs = np.ascontiguousarray(item_vecs[1:], dtype="float32")   # 去掉 padding 行
        dim = vecs.shape[1]
        if index_type == "hnsw":
            index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            index.hnsw.efSearch = ef_search
        else:
            index = faiss.IndexFlatIP(dim)        # 向量已 L2 归一化：内积 = 余弦
        index.add(vecs)
        return cls(index)

    def search(self, queries: np.ndarray, k: int):
        """返回 (scores, item_idx)，找不到的位置 item_idx 为 0。"""
        k = max(1, min(k, self.index.ntotal))
        queries = np.ascontiguousarray(queries, dtype="float32")
        scores, ids = self.index.search(queries, k)
        return scores, ids + 1

    def save(self, path) -> None:
        faiss.write_index(self.index, str(path))

    @classmethod
    def load(cls, path) -> "VectorIndex":
        return cls(faiss.read_index(str(path)))
