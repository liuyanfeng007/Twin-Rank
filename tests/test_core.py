"""不依赖 torch / faiss / lightgbm 的核心逻辑测试。"""
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from twin_rank.features.encoders import IdEncoder
from twin_rank.features.samples import build_history, pad_history
from twin_rank.ranking.features import FEATURE_NAMES, ItemTables, build_ranking_features
from twin_rank.ranking.metrics import metrics_from_positions, positions_by_score
from twin_rank.rerank.mmr import mmr_select
from twin_rank.rerank.reranker import Reranker
from twin_rank.retrieval.candidates import CandidateGenerator
from twin_rank.retrieval.popular import PopularRecall
from twin_rank.serving.catalog import ItemCatalog


class FakeIndex:
    """用 numpy 暴力内积代替 Faiss，接口与 VectorIndex.search 一致。"""
    def __init__(self, vecs):
        self.vecs = vecs[1:]
    def search(self, q, k):
        s = q @ self.vecs.T
        ids = np.argsort(-s, axis=1)[:, :k]
        return np.take_along_axis(s, ids, 1), ids + 1


def _vecs(n, d=8, seed=0):
    v = np.random.default_rng(seed).normal(size=(n, d)).astype("float32")
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_build_history_and_pad():
    users = np.array([1, 1, 1, 2, 2])
    items = np.array([10, 11, 12, 20, 21])
    hist = build_history(users, items, 2)
    assert hist.tolist() == [[0, 0], [0, 10], [10, 11], [0, 0], [0, 20]]
    assert pad_history(np.array([5, 6, 7]), 2).tolist() == [6, 7]
    assert pad_history(np.array([5]), 3).tolist() == [0, 0, 5]
    assert pad_history(np.array([], dtype=int), 3).tolist() == [0, 0, 0]


def test_popular_excludes_seen():
    pop = PopularRecall(np.array([3, 3, 3, 2, 2, 1]), n_items=5)
    assert pop.top(2).tolist() == [3, 2]
    assert pop.top(2, exclude=[3]).tolist() == [2, 1]


def test_candidates_exclude_seen_and_merge_popular():
    n_items, vecs = 30, _vecs(31)
    pop = PopularRecall(np.arange(1, 31).repeat(np.arange(30, 0, -1)), n_items=31)
    gen = CandidateGenerator(FakeIndex(vecs), vecs, pop, n_candidates=10, n_popular=3)
    seen = np.array([1, 2, 3, 4], dtype="int64")
    c = gen.generate(vecs[[5]], [seen])[0]
    assert len(c.item_idx) == 10 and len(set(c.item_idx)) == 10
    assert not set(c.item_idx) & set(seen)
    assert c.is_popular.sum() == 3 and (c.item_idx > 0).all()


def test_ranking_features_shapes_and_values():
    feats = np.zeros((6, 4), dtype="int64")
    feats[1:, 0] = [1, 1, 2, 2, 1]       # cat
    feats[1:, 1] = [1, 2, 1, 3, 1]       # store
    feats[1:, 2] = [3, 3, 5, 5, 4]       # price bucket
    tables = ItemTables(feats, np.log1p(np.arange(6)))
    hist = np.array([[0, 0, 2, 1], [0, 0, 0, 0]], dtype="int32")
    cands = [NS(item_idx=np.array([4, 5]), recall_score=np.array([.5, .9], "float32"), is_popular=np.array([0, 1])),
             NS(item_idx=np.array([3]), recall_score=np.array([.1], "float32"), is_popular=np.array([0]))]
    df = build_ranking_features(hist, cands, tables)
    assert len(df) == 3 and set(FEATURE_NAMES) <= set(df.columns)
    assert df["user_row"].tolist() == [0, 0, 1]
    assert df["recall_rank"].tolist() == [2, 1, 1]
    assert df.loc[0, "cat_match_cnt"] == 0 and df.loc[1, "cat_match_cnt"] == 2   # 物品5 类目1，历史两个都是类目1
    assert df.loc[0, "last_store_match"] == 0 and df.loc[1, "last_store_match"] == 1  # 最近物品1 的品牌 = 物品5 的品牌
    assert df.loc[2, "user_hist_len"] == 0


def test_metrics():
    user_row = np.array([0, 0, 0, 1, 1, 1, 2, 2])
    label = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    score = np.array([.9, .8, .1, .5, .4, .3, .2, .1])
    pos = positions_by_score(user_row, label, score, 3)
    assert pos.tolist()[:2] == [1, 2] and np.isinf(pos[2])
    m = metrics_from_positions(pos, [2, 3])
    assert abs(m["recall_at_2"] - 1 / 3) < 1e-9 and abs(m["recall_at_3"] - 2 / 3) < 1e-9


def test_mmr_diversity_and_store_cap():
    vecs = _vecs(7)
    vecs[2] = vecs[1]                              # 物品2 与物品1 完全相同
    items = np.array([1, 2, 3, 4, 5, 6])
    scores = np.array([1.0, 0.99, 0.5, 0.4, 0.3, 0.2])
    out = mmr_select(items, scores, vecs, 3, lam=0.5)
    assert 1 in out and 2 not in out               # 相同向量被打散
    stores = np.array([0, 7, 7, 7, 7, 8, 8])
    out = mmr_select(items, scores, vecs, 4, lam=1.0, store_codes=stores, max_same_store=2)
    assert (stores[out] == 7).sum() <= 2 and len(out) == 4


def test_reranker_blacklist_and_pin():
    enc = IdEncoder().fit(["a", "b", "c", "d", "e", "f"])
    cfg = NS(enabled=True, mmr_lambda=0.7, max_same_store=0, blacklist_items=["a"], pinned_items=["f"])
    rr = Reranker(cfg, _vecs(7), np.arange(7), enc)
    items = np.array([1, 2, 3, 4])
    out = rr.rerank(items, np.array([.9, .8, .7, .6]), 3)
    assert out[0] == enc.mapping["f"] and 1 not in out and len(out) == 3
    assert enc.mapping["f"] not in rr.rerank(items, np.array([.9, .8, .7, .6]), 3, exclude=[enc.mapping["f"]])


def test_catalog():
    enc = IdEncoder().fit(["x", "y", "z"])
    meta = pd.DataFrame({"item_id": ["x", "y"], "title": ["X", None], "store": ["S", None], "price": [9.99, np.nan]})
    cat = ItemCatalog(meta, enc)
    assert cat.info(1) == {"item_id": "x", "title": "X", "store": "S", "price": 9.99}
    assert cat.info(3)["title"] == "z" and cat.info(3)["price"] is None
    assert len(set(cat.store_codes[[2, 3]])) == 2   # 缺失品牌互不相同
