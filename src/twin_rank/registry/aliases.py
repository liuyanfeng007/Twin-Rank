"""MLflow 模型版本的别名与标签操作。

别名含义: champion = 线上在用；challenger = 新训练但未超过 champion 的候选；previous = 上一任 champion（回滚用）。
指标与评估设置以标签的形式存在每个版本上（metric.<名字>、fingerprint），所以随时能比较、能追溯。
"""
import json
from typing import Optional

from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def get_by_alias(client: MlflowClient, name: str, alias: str):
    try:
        return client.get_model_version_by_alias(name, alias)
    except MlflowException:
        return None


def write_tags(client: MlflowClient, name: str, version: str, evidence: dict) -> None:
    client.set_model_version_tag(name, version, "ranker_type", evidence["ranker_type"])
    client.set_model_version_tag(name, version, "fingerprint", json.dumps(evidence["fingerprint"], sort_keys=True))
    client.set_model_version_tag(name, version, "trained_at", evidence["trained_at"])
    for key, value in evidence["metrics"].items():
        client.set_model_version_tag(name, version, f"metric.{key}", f"{value:.6f}")


def read_evidence(model_version) -> Optional[tuple]:
    """从版本标签还原 (指标, 评估设置)；没有指标标签的老版本返回 None。"""
    tags = model_version.tags or {}
    metrics = {k[len("metric."):]: float(v) for k, v in tags.items() if k.startswith("metric.")}
    if not metrics or "fingerprint" not in tags:
        return None
    return metrics, json.loads(tags["fingerprint"])


def promote(client: MlflowClient, name: str, version, alias: str, previous_alias: str) -> None:
    """把 alias 指向 version，并把原来的 champion 留作 previous（用于一键回滚）。"""
    current = get_by_alias(client, name, alias)
    if current is not None and str(current.version) != str(version):
        client.set_registered_model_alias(name, previous_alias, str(current.version))
    client.set_registered_model_alias(name, alias, str(version))
