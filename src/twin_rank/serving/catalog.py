"""商品目录: item_idx -> 标题/品牌/价格，用于展示，以及重排时的"品牌打散"。

品牌缺失的商品各自独立编码，避免被误判成"同一品牌"而被限流。
"""
import numpy as np
import pandas as pd

from twin_rank.features.encoders import IdEncoder


class ItemCatalog:
    def __init__(self, items_meta: pd.DataFrame, item_enc: IdEncoder):
        n = len(item_enc)
        self.item_enc = item_enc
        self.item_ids = np.empty(n, dtype=object)
        for raw_id, idx in item_enc.mapping.items():
            self.item_ids[idx] = raw_id

        idx = item_enc.transform(items_meta["item_id"])
        ok = idx > 0
        self.titles = np.full(n, "", dtype=object)
        self.stores = np.full(n, None, dtype=object)
        self.prices = np.full(n, np.nan)
        self.titles[idx[ok]] = items_meta["title"].astype(object).where(items_meta["title"].notna(), "")[ok]
        self.stores[idx[ok]] = items_meta["store"].astype(object).where(items_meta["store"].notna(), None)[ok]
        self.prices[idx[ok]] = items_meta["price"].astype(float)[ok]

        # 品牌编码；品牌缺失的商品各自独立，避免被误判为"同一品牌"
        codes, _ = pd.factorize(self.stores)
        self.store_codes = np.where(codes >= 0, codes, -1 - np.arange(n))

    def info(self, idx: int) -> dict:
        price = self.prices[idx]
        return {
            "item_id": str(self.item_ids[idx]),
            "title": self.titles[idx] or str(self.item_ids[idx]),
            "store": self.stores[idx] or "",
            "price": None if np.isnan(price) else round(float(price), 2),
        }
