"""在线推荐主流程（RecommendService）。

recommend(): 用户历史 -> 用户向量 -> 多路召回 -> 精排特征 -> LightGBM 精排 -> MMR 重排 -> 返回，
并记录各阶段耗时。三种模式 recall / rank / full 见 constants.MODES。
也支持直接传入 history_item_ids，对任意用户或新会话做预测。
"""
import logging
import time
from typing import Optional

import numpy as np

from twin_rank.components import load_components
from twin_rank.constants import MODES  # noqa: F401  (api.py 从这里引用)
from twin_rank.features.samples import pad_history
from twin_rank.ranking.explain import top_reasons
from twin_rank.ranking.features import FEATURE_NAMES, build_ranking_features
from twin_rank.ranking.factory import load_ranker
from twin_rank.serving.cache import make_cache
from twin_rank.serving.history_store import HistoryStore

logger = logging.getLogger(__name__)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


class RecommendService:
    def __init__(self, cfg):
        self.cfg = cfg
        self.max_len = cfg.features.max_history_len
        self.stack = load_components(cfg)
        self.ranker = load_ranker(cfg, self.stack.tables)       # 由 ranker.model_type 决定
        self.history = HistoryStore(self.stack.encoded)
        self.cache = make_cache(cfg.serving.cache)
        logger.info("RecommendService 就绪: %d 用户, %d 商品",
                    len(self.history.seqs), self.stack.vocab["n_items"])

    # ---------- 对外接口 ----------
    def recommend(self, user_id: str, top_k: int = 10, mode: str = "full", explain: bool = False,
                  history_item_ids: Optional[list] = None) -> dict:
        """为一个用户生成推荐。

        参数:
            user_id: 原始用户 ID。不认识的用户 + 没传历史 => 走热门冷启动。
            top_k: 返回条数。
            mode: recall / rank / full，见 constants.MODES。
            explain: 是否计算每条推荐的理由（用 LightGBM 特征贡献，稍慢）。
            history_item_ids: 可选，直接指定该用户的历史商品（原始商品 ID，按时间从旧到新）。
                传入后不读取内部存储的历史，也不使用缓存，适合"给任意用户/新会话做预测"。
        返回: {"items": [...], "timings": {...}, "cold_start": bool, "cache_hit": bool, ...}
        """
        use_cache = history_item_ids is None
        key = f"rec:{user_id}:{mode}:{top_k}:{int(explain)}"
        if use_cache:
            cached = self.cache.get(key)
            if cached:
                return {**cached, "cache_hit": True}

        start, timings = time.perf_counter(), {}
        user_idx = self.stack.user_enc.mapping.get(user_id, 0)       # 0 = 未见过的用户
        seq = self._resolve_history(user_idx, history_item_ids)
        if len(seq) == 0:
            items, cold = self._cold_start(top_k), True
        else:
            items, cold = self._personalized(user_idx, seq, top_k, mode, explain, timings), False

        timings["total_ms"] = _ms(start)
        result = {"user_id": user_id, "mode": mode, "ranker": self.ranker.model_type, "cold_start": cold, "items": items,
                  "timings": timings, "cache_hit": False}
        if use_cache:
            self.cache.set(key, result)
        logger.info("recommend user=%s mode=%s %s", user_id, mode, timings)
        return result

    def _resolve_history(self, user_idx: int, history_item_ids: Optional[list]) -> np.ndarray:
        """取用户历史（item_idx 数组）。显式传入时把原始商品 ID 转成内部编号，丢弃未知商品。"""
        if history_item_ids is None:
            return self.history.get(user_idx)
        mapping = self.stack.item_enc.mapping
        return np.array([mapping[i] for i in history_item_ids if i in mapping], dtype="int64")

    def add_feedback(self, user_id: str, item_id: str) -> None:
        user_idx = self.stack.user_enc.mapping.get(user_id)
        item_idx = self.stack.item_enc.mapping.get(item_id)
        if not user_idx or not item_idx:
            raise KeyError("unknown user or item")
        self.history.append(user_idx, item_idx)
        self.cache.delete_prefix(f"rec:{user_id}:")

    def user_history(self, user_id: str, n: int = 10) -> list:
        seq = self.history.get(self.stack.user_enc.mapping.get(user_id, 0))
        return [self.stack.catalog.info(int(i)) for i in seq[-n:][::-1]]

    def sample_users(self, n: int) -> list:
        reverse = {idx: uid for uid, idx in self.stack.user_enc.mapping.items()}
        eligible = sorted(self.history.users_with_min_len(8))
        picked = np.random.default_rng(0).permutation(eligible)[:n]
        return [reverse[int(u)] for u in picked]

    # ---------- 内部流程 ----------
    def _cold_start(self, top_k: int) -> list:
        items = self.stack.popular.top(top_k)
        return [self._item(i, final_rank=r + 1, source="popular", reasons=["热门商品（新用户冷启动）"])
                for r, i in enumerate(items)]

    def _personalized(self, user_idx, seq, top_k, mode, explain, timings) -> list:
        hist = pad_history(seq, self.max_len)[None, :]

        t = time.perf_counter()
        vec = self.stack.user_encoder.encode(np.array([user_idx]), hist)
        timings["user_vector_ms"] = _ms(t)

        t = time.perf_counter()
        cands = self.stack.generator.generate(vec, [seq.astype("int64")])[0]
        timings["recall_ms"] = _ms(t)

        t = time.perf_counter()
        frame = build_ranking_features(hist, [cands], self.stack.tables)
        rank_scores = self.ranker.predict(frame, hist) if mode != "recall" else None
        timings["rank_ms"] = _ms(t)

        t = time.perf_counter()
        final = self._final_order(cands, rank_scores, seq, top_k, mode)
        timings["rerank_ms"] = _ms(t)

        return self._to_items(final, cands, frame, hist, rank_scores, explain)

    def _final_order(self, cands, rank_scores, seq, top_k, mode) -> np.ndarray:
        if mode == "recall":
            return cands.item_idx[np.argsort(-cands.recall_score)][:top_k]
        if mode == "rank":
            return cands.item_idx[np.argsort(-rank_scores)][:top_k]
        return self.stack.reranker.rerank(cands.item_idx, rank_scores, top_k, exclude=seq)

    def _to_items(self, final, cands, frame, hist, rank_scores, explain) -> list:
        row_of = {int(i): r for r, i in enumerate(cands.item_idx)}
        recall_pos = frame["recall_rank"].to_numpy().astype(int)
        rank_pos = None if rank_scores is None else (-rank_scores).argsort().argsort() + 1

        contrib = None
        rows = [row_of[int(i)] for i in final if int(i) in row_of]
        if explain and rank_scores is not None and rows:
            contrib = dict(zip(rows, self.ranker.contributions(frame.iloc[rows], hist)))

        items = []
        for pos, idx in enumerate(final, start=1):
            row = row_of.get(int(idx))
            if row is None:                                    # 运营置顶的商品，不在候选里
                items.append(self._item(idx, pos, source="pinned", reasons=["运营置顶"]))
                continue
            items.append(self._item(
                idx, pos,
                source="popular" if cands.is_popular[row] else "two_tower",
                recall_rank=int(recall_pos[row]),
                recall_score=float(cands.recall_score[row]),
                rank_position=None if rank_pos is None else int(rank_pos[row]),
                rank_score=None if rank_scores is None else float(rank_scores[row]),
                reasons=top_reasons(contrib[row]) if contrib else [],
            ))
        return items

    def _item(self, idx, final_rank, source, **extra) -> dict:
        base = {"recall_rank": None, "recall_score": None, "rank_position": None,
                "rank_score": None, "reasons": []}
        return {**self.stack.catalog.info(int(idx)), "final_rank": final_rank,
                "source": source, **base, **extra}
