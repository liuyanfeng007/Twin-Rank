"""物品侧特征：类目、品牌、价格桶、热度桶，整理成一张 [n_items, 4] 的整数查表。

查表随模型一起保存，所以训练和线上只需要传 item_idx，特征由模型内部查出来。
热度只用"训练集"统计，避免验证/测试信息泄露。价格缺失、品牌缺失分别落到 0 桶 / unknown。
"""
import numpy as np
import pandas as pd

from twin_rank.features.encoders import IdEncoder

FEATURE_COLUMNS = ["cat_idx", "store_idx", "price_bucket", "pop_bucket"]
VOCAB_KEYS = ["n_cat", "n_store", "n_price", "n_pop"]  # 与 FEATURE_COLUMNS 一一对应


def bucketize(values: pd.Series, n_buckets: int) -> np.ndarray:
    """按分位数分成 1..n_buckets 档，缺失值为 0。"""
    pct = values.rank(pct=True, method="first")
    buckets = np.ceil(pct * n_buckets).clip(1, n_buckets)
    return buckets.fillna(0).astype("int64").to_numpy()


def build_item_features(items_meta: pd.DataFrame, train: pd.DataFrame,
                        item_enc: IdEncoder, cfg):
    """
    返回 (特征表 ndarray[n_items, 4], 词表大小 dict)。
    train 须含 item_idx 列；热度只用训练集统计，避免泄露验证/测试信息。
    """
    df = pd.DataFrame({"item_id": list(item_enc.mapping.keys())})
    df["item_idx"] = item_enc.transform(df["item_id"])
    df = df.merge(items_meta[["item_id", "main_category", "store", "price"]],
                  on="item_id", how="left")

    category = df["main_category"].astype(object).where(df["main_category"].notna(), "unknown")
    cat_enc = IdEncoder().fit(category)
    df["cat_idx"] = cat_enc.transform(category)

    store = df["store"].astype(object).where(df["store"].notna(), "unknown")
    top_stores = store.value_counts().head(cfg.top_stores).index
    store = store.where(store.isin(top_stores), "other")
    store_enc = IdEncoder().fit(store)
    df["store_idx"] = store_enc.transform(store)

    df["price_bucket"] = bucketize(df["price"].astype(float), cfg.n_price_buckets)

    train_counts = train["item_idx"].value_counts()
    popularity = df["item_idx"].map(train_counts).fillna(0)
    df["pop_bucket"] = bucketize(popularity, cfg.n_pop_buckets)

    table = np.zeros((len(item_enc), len(FEATURE_COLUMNS)), dtype="int64")  # 第 0 行为 padding
    table[df["item_idx"].to_numpy()] = df[FEATURE_COLUMNS].to_numpy()

    vocab = {
        "n_cat": len(cat_enc),
        "n_store": len(store_enc),
        "n_price": cfg.n_price_buckets + 1,
        "n_pop": cfg.n_pop_buckets + 1,
    }
    return table, vocab
