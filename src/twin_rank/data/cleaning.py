"""交互数据清洗，输出统一的 [user_id, item_id, timestamp] 表。

步骤: 评分>=阈值视为正样本 -> 同一用户对同一商品只留最近一次 -> k-core 迭代过滤冷用户/冷商品。
k-core 是一个不动点过程：剔除冷商品会让部分用户变冷，反之亦然，所以循环直到不再变化。
同时负责把商品元数据整理成 [item_id, title, main_category, store, price]。
"""
import logging

import pandas as pd

logger = logging.getLogger(__name__)

REVIEW_COLUMNS = ["user_id", "parent_asin", "rating", "timestamp"]
META_COLUMNS = ["parent_asin", "title", "main_category", "store", "price"]


def keep_positive(df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
    df = df.dropna(subset=["user_id", "parent_asin", "rating", "timestamp"])
    return df[df["rating"] >= min_rating]


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """同一用户对同一商品只保留最近一次行为。"""
    df = df.sort_values("timestamp", kind="stable")
    return df.drop_duplicates(["user_id", "item_id"], keep="last")


def k_core_filter(df: pd.DataFrame, min_user: int, min_item: int) -> pd.DataFrame:
    """迭代剔除冷用户/冷商品，直到都满足阈值。"""
    while True:
        before = len(df)
        item_cnt = df["item_id"].value_counts()
        df = df[df["item_id"].map(item_cnt) >= min_item]
        user_cnt = df["user_id"].value_counts()
        df = df[df["user_id"].map(user_cnt) >= min_user]
        if len(df) == before:
            return df


def build_interactions(reviews: pd.DataFrame, cfg) -> pd.DataFrame:
    df = keep_positive(reviews, cfg.positive_rating)
    df = df.rename(columns={"parent_asin": "item_id"})
    df = deduplicate(df)
    df = k_core_filter(df, cfg.min_user_interactions, cfg.min_item_interactions)
    logger.info(
        "清洗后: %d 条交互, %d 用户, %d 商品",
        len(df), df["user_id"].nunique(), df["item_id"].nunique(),
    )
    return df[["user_id", "item_id", "timestamp"]].reset_index(drop=True)


def clean_items_meta(meta: pd.DataFrame, item_ids) -> pd.DataFrame:
    meta = meta.rename(columns={"parent_asin": "item_id"})
    meta = meta[meta["item_id"].isin(set(item_ids))].copy()
    meta = meta.drop_duplicates("item_id")
    meta["price"] = pd.to_numeric(meta["price"], errors="coerce")
    for col in ("title", "main_category", "store"):
        meta[col] = meta[col].astype("string")
    return meta.reset_index(drop=True)
