"""自定义数据接入与注册模型输入输出的测试（不依赖 torch / mlflow / faiss / lightgbm）。"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from twin_rank.data.cleaning import build_interactions, clean_items_meta
from twin_rank.data.custom import load_custom_meta, load_custom_reviews
from twin_rank.data.split import assign_split
from twin_rank.registry.io import OUTPUT_COLUMNS, flatten_response, parse_requests


def _custom_cfg(tmp, with_items=True, rating="rating"):
    return NS(
        interactions_file=str(tmp / "interactions.csv"),
        items_file=str(tmp / "items.csv") if with_items else None,
        columns=NS(user_id="user_id", item_id="item_id", timestamp="timestamp", rating=rating),
        item_columns=NS(item_id="item_id", title="title", category="category", store="brand", price="price"),
    )


def _generate(tmp):
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "tools" / "make_demo_custom_data.py"), "--out", str(tmp),
                    "--users", "300", "--items", "200"], check=True, capture_output=True)


def test_custom_data_end_to_end(tmp_path):
    _generate(tmp_path)
    cfg = _custom_cfg(tmp_path)
    reviews = load_custom_reviews(cfg)
    assert list(reviews.columns) == ["user_id", "parent_asin", "rating", "timestamp"]
    assert reviews["timestamp"].notna().all()                       # 日期字符串被转成可排序数值
    inter = assign_split(build_interactions(reviews, NS(positive_rating=4.0, min_user_interactions=5,
                                                        min_item_interactions=5)))
    assert set(inter["split"]) == {"train", "val", "test"}
    last = inter.groupby("user_id").tail(1)
    assert (last["split"] == "test").all()                          # 每个用户最后一次交互是测试
    meta = clean_items_meta(load_custom_meta(cfg), inter["item_id"].unique())
    assert {"title", "store", "price"} <= set(meta.columns) and len(meta) > 0


def test_custom_data_without_items_or_rating(tmp_path):
    _generate(tmp_path)
    cfg = _custom_cfg(tmp_path, with_items=False, rating=None)
    reviews = load_custom_reviews(cfg)
    assert (reviews["rating"] == 5.0).all()                         # 没有评分列：全部视为正样本
    assert load_custom_meta(cfg).empty
    meta = clean_items_meta(load_custom_meta(cfg), ["I00001"])
    assert len(meta) == 0


def test_custom_data_missing_column_message(tmp_path):
    _generate(tmp_path)
    cfg = _custom_cfg(tmp_path)
    cfg.columns.user_id = "no_such_column"
    try:
        load_custom_reviews(cfg)
        raise AssertionError("应当报错")
    except ValueError as exc:
        assert "no_such_column" in str(exc)


def test_parse_requests_defaults_and_history():
    df = pd.DataFrame({"user_id": ["u1", "u2", "u3"], "top_k": [5, np.nan, 3],
                       "history": ["a, b", None, ["x", "y"]]})
    reqs = parse_requests(df)
    assert [r.top_k for r in reqs] == [5, 10, 3]
    assert reqs[0].history == ["a", "b"] and reqs[1].history is None and reqs[2].history == ["x", "y"]
    assert all(r.mode == "full" and r.explain is False for r in reqs)


def test_parse_requests_validation():
    for bad in (pd.DataFrame({"top_k": [1]}), pd.DataFrame({"user_id": ["u"], "mode": ["bogus"]})):
        try:
            parse_requests(bad)
            raise AssertionError("应当报错")
        except ValueError:
            pass


def test_flatten_response():
    result = {"user_id": "u1", "cold_start": False, "items": [{
        "item_id": "A", "title": "t", "store": "s", "price": 1.0, "final_rank": 1, "source": "two_tower",
        "recall_rank": 3, "recall_score": 0.5, "rank_score": 0.9, "reasons": ["r1", "r2"]}]}
    df = flatten_response(result)
    assert list(df.columns) == OUTPUT_COLUMNS and df.loc[0, "reasons"] == "r1；r2"
    assert len(flatten_response({"user_id": "u", "cold_start": True, "items": []})) == 0
