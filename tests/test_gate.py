"""质量门禁与训练证据的测试（不依赖 mlflow / torch）。"""
import json
from pathlib import Path

import numpy as np

from twin_rank.config import load_config
from twin_rank.registry.evidence import (data_fingerprint, evidence_path, load_verified_evidence,
                                         save_evidence)
from twin_rank.registry.gate import decide

M = "test_full_ndcg_at_10"
FP = {"source": "amazon", "dataset": "Video_Games", "n_users": 100, "n_items": 50,
      "eval_users": 1000, "eval_ks": [10, 20, 50], "seed": 42}


def test_first_version_is_promoted():
    d = decide(M, {M: 0.03}, FP, None, None)
    assert d.promote and d.comparable


def test_better_version_promoted_worse_not():
    assert decide(M, {M: 0.035}, FP, {M: 0.030}, FP).promote
    worse = decide(M, {M: 0.025}, FP, {M: 0.030}, FP)
    assert not worse.promote and worse.comparable and "0.0250" in worse.reason


def test_equal_or_within_margin_not_promoted():
    assert not decide(M, {M: 0.030}, FP, {M: 0.030}, FP).promote                       # 持平不晋升
    assert not decide(M, {M: 0.0305}, FP, {M: 0.030}, FP, min_improvement=0.001).promote   # 噪声范围内
    assert decide(M, {M: 0.032}, FP, {M: 0.030}, FP, min_improvement=0.001).promote


def test_incomparable_never_auto_promoted():
    other = {**FP, "dataset": "Digital_Music"}
    d = decide(M, {M: 0.99}, FP, {M: 0.01}, other)           # 指标再高，口径不同也不能自动晋升
    assert not d.promote and not d.comparable
    d = decide(M, {M: 0.99}, FP, {"other_metric": 0.1}, FP)  # champion 没有该指标
    assert not d.promote and not d.comparable


def test_missing_metric_is_reported():
    try:
        decide("nope", {M: 0.1}, FP, None, None)
        raise AssertionError("应当报错")
    except ValueError as exc:
        assert M in str(exc)


def _cfg(tmp: Path):
    cfg, _ = load_config("configs/config.yaml")
    cfg.data.artifact_dir = str(tmp)
    (tmp / cfg.ranker.model_name).write_text("model-v1")
    np.save(tmp / "item_embeddings.npy", np.zeros((3, 2)))
    return cfg


def test_evidence_roundtrip_and_stale_detection(tmp_path):
    cfg = _cfg(tmp_path)
    fp = data_fingerprint(cfg, {"n_users": 10, "n_items": 5}, 100)
    save_evidence(cfg, {M: 0.04}, fp)
    evidence = load_verified_evidence(cfg)
    assert evidence["metrics"][M] == 0.04 and evidence["fingerprint"] == fp and evidence["ranker_type"] == "lightgbm"

    (tmp_path / cfg.ranker.model_name).write_text("model-v2")   # 精排被重训，但指标还是旧的
    try:
        load_verified_evidence(cfg)
        raise AssertionError("应当报错")
    except RuntimeError as exc:
        assert "train_ranker" in str(exc)


def test_missing_evidence_message(tmp_path):
    cfg = _cfg(tmp_path)
    assert not evidence_path(cfg).exists()
    try:
        load_verified_evidence(cfg)
        raise AssertionError("应当报错")
    except FileNotFoundError as exc:
        assert "train_ranker" in str(exc)
