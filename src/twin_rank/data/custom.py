"""读取用户自己的数据，转成与 Amazon 数据相同的标准列，后续清洗/切分/特征流程完全复用。

行为表必需列：用户 ID、商品 ID、时间戳；可选列：评分（没有就视为全部是正样本）。
商品表可选；缺失的列（标题/类目/品牌/价格）会被忽略，对应特征退化为"未知"。
"""
from pathlib import Path

import numpy as np
import pandas as pd

from twin_rank.data.cleaning import META_COLUMNS, REVIEW_COLUMNS


def _read_table(path) -> pd.DataFrame:
    """按后缀读取 csv / tsv / parquet / jsonl / json（csv 可带 .gz 压缩）。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件: {path}（请检查 data.custom 配置）")
    name = path.name.lower()
    if name.endswith(".parquet"):
        return pd.read_parquet(path)
    if name.endswith((".jsonl", ".jsonl.gz")):
        return pd.read_json(path, lines=True)
    if name.endswith(".json"):
        return pd.read_json(path)
    if name.endswith((".tsv", ".tsv.gz")):
        return pd.read_csv(path, sep="\t")
    return pd.read_csv(path)


def _to_timestamp(series: pd.Series) -> pd.Series:
    """数字时间戳原样使用；日期字符串转成毫秒数。只用于排序，所以单位无所谓。"""
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.9:
        return numeric
    dt = pd.to_datetime(series, errors="coerce")
    millis = dt.astype("int64") / 1e6
    return pd.Series(np.where(dt.isna(), np.nan, millis), index=series.index)


def _column(df: pd.DataFrame, name):
    """取可选列；名字为空或文件里没有该列时返回 None。"""
    return df[name] if name and name in df.columns else None


def load_custom_reviews(custom) -> pd.DataFrame:
    """返回标准行为表 [user_id, parent_asin, rating, timestamp]（parent_asin 即商品 ID）。"""
    cols = vars(custom.columns)
    df = _read_table(custom.interactions_file)
    required = [cols["user_id"], cols["item_id"], cols["timestamp"]]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"行为表缺少列 {missing}；文件现有列: {list(df.columns)}。请检查 data.custom.columns 映射")

    rating_col = _column(df, cols.get("rating"))
    out = pd.DataFrame({
        # ID 统一转字符串：编码器以 JSON 保存，数字 ID 会在加载后变成字符串而对不上
        "user_id": df[cols["user_id"]].astype(str),
        "parent_asin": df[cols["item_id"]].astype(str),
        "rating": pd.to_numeric(rating_col, errors="coerce") if rating_col is not None else 5.0,
        "timestamp": _to_timestamp(df[cols["timestamp"]]),
    })
    return out[REVIEW_COLUMNS]


def load_custom_meta(custom) -> pd.DataFrame:
    """返回标准商品表 [parent_asin, title, main_category, store, price]；未提供商品表时返回空表。"""
    if not custom.items_file:
        return pd.DataFrame(columns=META_COLUMNS)
    cols = vars(custom.item_columns)
    df = _read_table(custom.items_file)
    if cols["item_id"] not in df.columns:
        raise ValueError(f"商品表缺少商品 ID 列 {cols['item_id']!r}；现有列: {list(df.columns)}")

    out = pd.DataFrame({
        "parent_asin": df[cols["item_id"]].astype(str),
        "title": _column(df, cols.get("title")),
        "main_category": _column(df, cols.get("category")),
        "store": _column(df, cols.get("store")),
        "price": _column(df, cols.get("price")),
    })
    return out[META_COLUMNS]
