"""注册模型的输入输出格式转换（纯 pandas/numpy，不依赖 mlflow，方便单测）。

输入 DataFrame（每行一个预测请求）:
    user_id  必填  原始用户 ID
    top_k    可选  默认 10
    mode     可选  recall / rank / full，默认 full
    explain  可选  是否返回推荐理由，默认 False
    history  可选  历史商品 ID 列表（或逗号分隔字符串）。提供后忽略库内历史，可对任意用户/新会话预测

输出 DataFrame（每个推荐一行）列见 OUTPUT_COLUMNS。
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from twin_rank.constants import MODES

OUTPUT_COLUMNS = ["user_id", "final_rank", "item_id", "title", "store", "price", "source",
                  "recall_rank", "recall_score", "rank_score", "reasons", "cold_start"]


@dataclass
class PredictRequest:
    user_id: str
    top_k: int = 10
    mode: str = "full"
    explain: bool = False
    history: Optional[list] = None


def _scalar(row: pd.Series, name: str, default):
    """取一个标量字段；列不存在或为空时用默认值。"""
    if name not in row.index:
        return default
    value = row[name]
    return default if value is None or (np.isscalar(value) and pd.isna(value)) else value


def _history(row: pd.Series) -> Optional[list]:
    """history 可能是 list / ndarray / 逗号分隔字符串 / 缺失。None 表示"使用库内历史"。"""
    if "history" not in row.index:
        return None
    value = row["history"]
    if value is None or (np.isscalar(value) and not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, str):
        items = [s.strip() for s in value.split(",") if s.strip()]
        return items or None
    return [str(x) for x in value]


def parse_requests(df: pd.DataFrame) -> list:
    if "user_id" not in df.columns:
        raise ValueError("输入必须包含 user_id 列，可选列: top_k / mode / explain / history")
    requests = []
    for _, row in df.iterrows():
        mode = str(_scalar(row, "mode", "full"))
        if mode not in MODES:
            raise ValueError(f"mode 必须是 {MODES} 之一，收到 {mode!r}")
        requests.append(PredictRequest(
            user_id=str(row["user_id"]),
            top_k=int(_scalar(row, "top_k", 10)),
            mode=mode,
            explain=bool(_scalar(row, "explain", False)),
            history=_history(row),
        ))
    return requests


def flatten_response(result: dict) -> pd.DataFrame:
    """把 RecommendService.recommend 的返回值展开成一行一个推荐。"""
    rows = [{
        "user_id": result["user_id"],
        "final_rank": item["final_rank"],
        "item_id": item["item_id"],
        "title": item["title"],
        "store": item["store"],
        "price": item["price"],
        "source": item["source"],
        "recall_rank": item["recall_rank"],
        "recall_score": item["recall_score"],
        "rank_score": item["rank_score"],
        "reasons": "；".join(item["reasons"]),
        "cold_start": result["cold_start"],
    } for item in result["items"]]
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
