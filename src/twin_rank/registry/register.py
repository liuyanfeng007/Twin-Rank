"""把当前训练好的整套系统注册到 MLflow Model Registry，并按"质量门禁"决定是否晋升 champion。

流程：每次都注册成新版本（保留完整历史）并写上指标标签 -> 与当前 champion 比较 ->
  * 更好：champion 指向新版本，旧 champion 留作 previous；
  * 没有更好 / 不可比：新版本只打 challenger 别名，线上继续用旧 champion。
这样"最新 ≠ 最好"，不会因为一次变差的训练就把线上模型换掉。
"""
import logging
import tempfile
from pathlib import Path

import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient

import twin_rank
from twin_rank.registry import aliases
from twin_rank.registry.bundle import build_bundle
from twin_rank.registry.evidence import load_verified_evidence
from twin_rank.registry.gate import Decision, decide
from twin_rank.registry.pyfunc_model import TwinRankModel
from twin_rank.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)

# 预测环境需要的依赖（MLflow 会写进模型的 requirements.txt，用 mlflow models serve 时据此建环境）
PIP_REQUIREMENTS = ["numpy", "pandas", "pyarrow", "scipy", "pyyaml", "torch",
                    "faiss-cpu", "lightgbm", "mlflow"]


def _log_pyfunc(bundle: Path, model_name: str):
    """兼容 MLflow 2 / 3：artifact_path→name、code_path→code_paths 的参数名变化。"""
    base = dict(python_model=TwinRankModel(), artifacts={"bundle": str(bundle)},
                pip_requirements=PIP_REQUIREMENTS, registered_model_name=model_name)
    code_dir = str(Path(twin_rank.__file__).parent)          # 把 twin_rank 源码一起打进模型，加载时无需另装
    attempts = [dict(name="model", code_paths=[code_dir]),
                dict(artifact_path="model", code_paths=[code_dir]),
                dict(artifact_path="model", code_path=[code_dir])]
    for extra in attempts:
        try:
            return mlflow.pyfunc.log_model(**base, **extra)
        except TypeError:
            continue
    raise RuntimeError("当前 MLflow 版本的 pyfunc.log_model 参数不兼容，请升级 mlflow>=2.9")


def register_model(cfg, raw_cfg: dict, alias: str = None, force: bool = False) -> dict:
    """
    alias: 显式指定别名（如 staging）时，直接打该别名，不做门禁比较。
    force: 无视比较结果，强制晋升为 champion。
    返回 {"name", "version", "status"}，status 为 promoted / challenger / <显式别名>。
    """
    reg = cfg.registry
    evidence = load_verified_evidence(cfg)          # 指标必须对应当前模型文件，否则报错
    setup_mlflow(cfg)
    name = reg.model_name

    with tempfile.TemporaryDirectory() as tmp:
        bundle = build_bundle(cfg, raw_cfg, Path(tmp) / "bundle")
        with mlflow.start_run(run_name=reg.run_name):
            mlflow.set_tag("ranker_type", evidence["ranker_type"])
            mlflow.log_params({"data_source": cfg.data.source, "n_candidates": cfg.retrieval.n_candidates,
                               "index_type": cfg.retrieval.index_type, "ranker_type": evidence["ranker_type"]})
            mlflow.log_metrics(evidence["metrics"])
            _log_pyfunc(bundle, name)

    client = MlflowClient()
    version = str(max(int(v.version) for v in client.search_model_versions(f"name='{name}'")))
    aliases.write_tags(client, name, version, evidence)

    if alias:                                        # 人工指定：不比较
        client.set_registered_model_alias(name, alias, version)
        logger.info("已注册 %s 版本 %s，按指定别名 @%s", name, version, alias)
        return {"name": name, "version": version, "status": alias}

    champion = aliases.get_by_alias(client, name, reg.alias)
    champion_evidence = aliases.read_evidence(champion) if champion is not None else None
    if champion is not None and champion_evidence is None:
        # champion 存在但没有指标标签（旧流程注册的）：无法比较，不能当作"没有 champion"
        decision = Decision(False, False, f"当前 champion(v{champion.version}) 没有指标标签（旧流程注册的），无法比较")
    else:
        decision = decide(reg.promote_metric, evidence["metrics"], evidence["fingerprint"],
                          champion_evidence[0] if champion_evidence else None,
                          champion_evidence[1] if champion_evidence else None,
                          reg.min_improvement)

    if decision.promote or force:
        aliases.promote(client, name, version, reg.alias, reg.previous_alias)
        why = "（--force 强制晋升）" if force and not decision.promote else f"（{decision.reason}）"
        logger.info("✔ 版本 %s 晋升为 @%s %s", version, reg.alias, why)
        return {"name": name, "version": version, "status": "promoted"}

    client.set_registered_model_alias(name, reg.challenger_alias, version)
    logger.warning("✘ 版本 %s 未晋升，线上仍是 @%s(v%s)。原因: %s", version, reg.alias,
                   champion.version if champion else "-", decision.reason)
    logger.warning("  已打上 @%s，可用 models:/%s@%s 单独试用；确认要上线请运行: "
                   "python -m twin_rank.pipelines.promote_model --version %s",
                   reg.challenger_alias, name, reg.challenger_alias, version)
    return {"name": name, "version": version, "status": "challenger"}
