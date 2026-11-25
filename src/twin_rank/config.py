"""读取 YAML 配置。

load_config 返回两份：
  * cfg  —— 属性访问版（cfg.train.lr），代码里使用；
  * raw  —— 原始 dict，用于整体写入 MLflow 参数、或打包进注册模型。
"""
import os
from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    return obj


def load_config(path: str = "configs/config.yaml"):
    """返回 (属性访问版配置, 原始 dict)。原始 dict 用于写入 MLflow 参数。"""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    # 环境变量临时切换精排模型，便于对比实验：TWIN_RANK_RANKER=din python -m ...train_ranker
    override = os.environ.get("TWIN_RANK_RANKER")
    if override:
        raw.setdefault("ranker", {})["model_type"] = override
    return _to_namespace(raw), raw
