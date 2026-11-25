"""精排模型选择相关的测试（不依赖 torch / lightgbm）。"""
import os
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from twin_rank.config import load_config
from twin_rank.constants import RANKER_TYPES
from twin_rank.paths import ranker_path, ranker_type
from twin_rank.ranking.deep.features import DENSE_FEATURES, LOG_FEATURES, Scaler, raw_dense
from twin_rank.ranking.factory import build_ranker, load_ranker
from twin_rank.ranking.features import FEATURE_NAMES

ROOT_CFG = "configs/config.yaml"


def test_dense_features_are_subset_of_ranking_features():
    assert set(DENSE_FEATURES) <= set(FEATURE_NAMES) and LOG_FEATURES <= set(DENSE_FEATURES)


def test_raw_dense_and_scaler_roundtrip():
    frame = pd.DataFrame({n: np.arange(1, 6, dtype="float32") for n in DENSE_FEATURES})
    raw = raw_dense(frame)
    assert raw.shape == (5, len(DENSE_FEATURES))
    assert np.isclose(raw[0, DENSE_FEATURES.index("recall_rank")], np.log1p(1.0))   # 计数类做了 log1p
    scaler = Scaler.fit(raw)
    z = scaler.transform(raw)
    assert np.allclose(z.mean(axis=0), 0, atol=1e-5) and np.allclose(z.std(axis=0), 1, atol=1e-3)
    again = Scaler.from_state(scaler.state()).transform(raw)                         # 存取后结果一致
    assert np.allclose(z, again)


def test_ranker_path_per_type():
    cfg, _ = load_config(ROOT_CFG)
    assert ranker_type(cfg) == "lightgbm" and ranker_path(cfg).name == "lightgbm_ranker.txt"
    cfg.ranker.model_type = "din"
    assert ranker_path(cfg).name == "deep_ranker_din.pt"
    cfg.ranker.model_type = "dcn"
    assert ranker_path(cfg).name == "deep_ranker_dcn.pt"     # 不同类型各存各的，互不覆盖


def test_env_override_and_invalid_type():
    os.environ["TWIN_RANK_RANKER"] = "deepfm"
    try:
        cfg, raw = load_config(ROOT_CFG)
    finally:
        del os.environ["TWIN_RANK_RANKER"]
    assert cfg.ranker.model_type == "deepfm" and raw["ranker"]["model_type"] == "deepfm"

    cfg.ranker.model_type = "xgboost"
    for fn in (lambda: build_ranker(cfg, None), lambda: load_ranker(cfg, None)):
        try:
            fn()
            raise AssertionError("应当报错")
        except ValueError as exc:
            assert "xgboost" in str(exc) and "deepfm" in str(exc)


def test_missing_model_file_message():
    cfg, _ = load_config(ROOT_CFG)
    cfg.data.artifact_dir = "/nonexistent_dir"
    try:
        load_ranker(cfg, None)
        raise AssertionError("应当报错")
    except FileNotFoundError as exc:
        assert "train_ranker" in str(exc)


def test_all_ranker_types_listed():
    assert RANKER_TYPES == ("lightgbm", "deepfm", "dcn", "din")
