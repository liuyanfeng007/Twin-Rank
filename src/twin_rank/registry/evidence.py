"""训练证据：train_ranker 结束时把"这次评估的指标 + 评估设置 + 模型文件指纹"存成 json，
register_model 注册时读取它，作为是否晋升 champion 的依据。

为什么不直接去 MLflow 里查最近一次 run？因为不同精排模型会产生多个 run，容易取错；
证据文件和模型文件放在同一目录，并带文件指纹，能保证"指标"和"被注册的模型"是同一份。
"""
import json
import time
from pathlib import Path

from twin_rank.paths import artifact_dir, item_embeddings_path, ranker_path, ranker_type
from twin_rank.utils.hashing import sha256_file


def evidence_path(cfg) -> Path:
    return artifact_dir(cfg) / f"ranker_metrics_{ranker_type(cfg)}.json"


def data_fingerprint(cfg, vocab: dict, n_eval_users: int) -> dict:
    """评估口径：数据、测试用户数、K 值、随机种子都相同，不同版本的指标才可比较。"""
    dataset = cfg.data.category if cfg.data.source == "amazon" else str(cfg.data.custom.interactions_file)
    return {"source": cfg.data.source, "dataset": dataset, "n_users": vocab["n_users"],
            "n_items": vocab["n_items"], "eval_users": int(n_eval_users),
            "eval_ks": list(cfg.ranker.eval_ks), "seed": cfg.train.seed}


def save_evidence(cfg, metrics: dict, fingerprint: dict) -> Path:
    """metrics 的键与 MLflow 里一致，如 test_full_ndcg_at_10。"""
    payload = {
        "ranker_type": ranker_type(cfg),
        "metrics": {k: float(v) for k, v in metrics.items()},
        "fingerprint": fingerprint,
        "ranker_sha256": sha256_file(ranker_path(cfg)),
        "embeddings_sha256": sha256_file(item_embeddings_path(cfg)),
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    path = evidence_path(cfg)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_verified_evidence(cfg) -> dict:
    """读取证据并校验它对应的就是当前的精排模型与物品向量；不一致说明模型被重训过而指标是旧的。"""
    path = evidence_path(cfg)
    if not path.exists():
        raise FileNotFoundError(f"找不到训练证据 {path}。请先运行 train_ranker（ranker.model_type={ranker_type(cfg)}）")
    evidence = json.loads(path.read_text(encoding="utf-8"))
    stale = (evidence["ranker_sha256"] != sha256_file(ranker_path(cfg))
             or evidence["embeddings_sha256"] != sha256_file(item_embeddings_path(cfg)))
    if stale:
        raise RuntimeError("训练证据与当前模型文件不一致（双塔或精排在评估之后又被重新训练过）。"
                           "请重新运行 train_ranker，再注册")
    return evidence
