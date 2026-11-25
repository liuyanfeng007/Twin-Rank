"""MMR（Maximal Marginal Relevance）多样性重排。

每一步选 lam*相关性 - (1-lam)*与已选商品的最大相似度 最高者：lam 越小越强调多样性。
可附加"同一品牌最多 N 个"的硬约束；约束过严导致不足 top_k 时，按相关性补齐。
"""

import numpy as np


def mmr_select(items: np.ndarray, scores: np.ndarray, item_vecs: np.ndarray, top_k: int,
               lam: float, store_codes: np.ndarray = None, max_same_store: int = 0) -> np.ndarray:
    """
    MMR 多样性重排：每一步选 lam*相关性 - (1-lam)*与已选物品的最大相似度 最高者。
    max_same_store > 0 时，同一品牌最多入选这么多个。
    """
    n = len(items)
    top_k = min(top_k, n)
    rel = (scores - scores.min()) / (np.ptp(scores) + 1e-9)
    vecs = item_vecs[items]
    sim = vecs @ vecs.T

    remaining = np.ones(n, dtype=bool)
    store_full = np.zeros(n, dtype=bool)
    store_count, max_sim, selected = {}, np.zeros(n), []

    for _ in range(top_k):
        mmr = lam * rel - (1.0 - lam) * max_sim
        mmr = np.where(~remaining | store_full, -np.inf, mmr)
        j = int(mmr.argmax())
        if not np.isfinite(mmr[j]):
            break
        selected.append(j)
        remaining[j] = False
        max_sim = np.maximum(max_sim, sim[j])
        if max_same_store and store_codes is not None:
            code = store_codes[items[j]]
            store_count[code] = store_count.get(code, 0) + 1
            if store_count[code] >= max_same_store:
                store_full |= store_codes[items] == code

    if len(selected) < top_k:                    # 约束过严导致不足，按相关性补齐
        rest = [j for j in np.argsort(-rel) if remaining[j]]
        selected += rest[: top_k - len(selected)]
    return items[selected]
