"""重排与业务规则，按顺序: 黑名单过滤 -> MMR 打散 + 品牌上限 -> 运营置顶。

配置见 config.yaml 的 rerank.*；置顶商品若用户已看过则自动跳过。
"""
import numpy as np

from twin_rank.rerank.mmr import mmr_select


class Reranker:
    def __init__(self, cfg, item_vecs: np.ndarray, store_codes: np.ndarray, item_enc):
        self.enabled = cfg.enabled
        self.lam = cfg.mmr_lambda
        self.max_same_store = cfg.max_same_store
        self.item_vecs = item_vecs
        self.store_codes = store_codes
        to_idx = lambda ids: [item_enc.mapping[i] for i in ids if i in item_enc.mapping]
        self.blacklist = np.array(to_idx(cfg.blacklist_items), dtype="int64")
        self.pinned = to_idx(cfg.pinned_items)

    def rerank(self, item_idx: np.ndarray, scores: np.ndarray, top_k: int, exclude=()) -> np.ndarray:
        keep = ~np.isin(item_idx, self.blacklist)
        items, scores = item_idx[keep], scores[keep]

        if self.enabled and len(items) > 0:
            ordered = mmr_select(items, scores, self.item_vecs, top_k, self.lam,
                                 self.store_codes, self.max_same_store)
        else:
            ordered = items[np.argsort(-scores)][:top_k]

        exclude = set(int(i) for i in exclude)
        pins = [p for p in self.pinned if p not in exclude and p not in set(self.blacklist.tolist())]
        final = pins + [int(i) for i in ordered if int(i) not in pins]
        return np.array(final[:top_k], dtype="int64")
