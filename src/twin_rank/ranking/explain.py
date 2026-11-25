"""把 LightGBM 的特征贡献(pred_contrib)翻译成人能看懂的推荐理由。
"""
import numpy as np

from twin_rank.ranking.features import FEATURE_NAMES

REASON_TEXT = {
    "recall_score": "与你的兴趣向量高度相似",
    "recall_rank": "在兴趣召回中排名靠前",
    "cat_match_cnt": "与你浏览过的类目一致",
    "store_match_cnt": "来自你常看的品牌",
    "last_store_match": "与你最近一次行为同品牌",
    "price_diff": "价格接近你的偏好",
    "pop_bucket": "近期较受欢迎",
    "item_log_pop": "整体热度较高",
    "is_popular": "热门补充推荐",
}


def top_reasons(contrib_row: np.ndarray, k: int = 2) -> list:
    """contrib_row 长度 = 特征数 + 1（bias）。取正向贡献最大的 k 个有文案的特征。"""
    contributions = contrib_row[: len(FEATURE_NAMES)]
    reasons = []
    for idx in np.argsort(-contributions):
        name = FEATURE_NAMES[idx]
        if contributions[idx] <= 0 or name not in REASON_TEXT:
            continue
        reasons.append(REASON_TEXT[name])
        if len(reasons) == k:
            break
    return reasons
