"""步骤 6：构造精排样本、训练精排模型（LightGBM / DeepFM / DCN / DIN）、做端到端评估并记录到 MLflow。

用哪个模型由 config.yaml 的 ranker.model_type 决定，也可临时覆盖：
    TWIN_RANK_RANKER=din python -m twin_rank.pipelines.train_ranker
输出: data/artifacts/ 下的精排模型文件（lightgbm_ranker.txt 或 deep_ranker_<类型>.pt）、feature_importance_<类型>.csv
"""
import logging

import mlflow
import numpy as np

from twin_rank.components import load_components
from twin_rank.paths import feature_dir, ranker_path, ranker_type
from twin_rank.pipelines._cli import init
from twin_rank.ranking.dataset import build_ranker_frame, drop_groups_without_positive, take_users
from twin_rank.ranking.evaluate import evaluate_variants
from twin_rank.ranking.factory import build_ranker
from twin_rank.registry.evidence import data_fingerprint, save_evidence
from twin_rank.training.data import build_seen_matrix, load_arrays
from twin_rank.utils.common import set_seed
from twin_rank.utils.mlflow_utils import flatten_dict, setup_mlflow

logger = logging.getLogger(__name__)


def main():
    cfg, raw_cfg = init("训练精排模型并做端到端评估")
    set_seed(cfg.train.seed)
    rng = np.random.default_rng(cfg.train.seed)
    rk, kind = cfg.ranker, ranker_type(cfg)

    stack = load_components(cfg)
    feat_dir = feature_dir(cfg)
    val, test = load_arrays(feat_dir / "val.npz"), load_arrays(feat_dir / "test.npz")
    val_seen = build_seen_matrix(stack.encoded, ["train"], stack.vocab)
    test_seen = build_seen_matrix(stack.encoded, ["train", "val"], stack.vocab)

    # 精排样本来自验证集用户（双塔没在这些目标上训练过，分布更接近线上）
    rows = rng.permutation(len(val["user_idx"]))[: rk.max_train_users]
    cut = int(len(rows) * (1 - rk.valid_ratio))
    train_sub, valid_sub = take_users(val, rows[:cut]), take_users(val, rows[cut:])
    train_df = build_ranker_frame(train_sub, val_seen, stack)
    valid_df = build_ranker_frame(valid_sub, val_seen, stack)
    n_train_users, n_valid_users = train_df["user_row"].nunique(), valid_df["user_row"].nunique()
    train_df, valid_df = drop_groups_without_positive(train_df), drop_groups_without_positive(valid_df)
    logger.info("精排样本: train %d/%d 个用户的候选含目标, valid %d/%d（仅保留这些用户训练）",
                train_df["user_row"].nunique(), n_train_users, valid_df["user_row"].nunique(), n_valid_users)

    test_rows = rng.permutation(len(test["user_idx"]))[: rk.eval_users]
    test_sub = take_users(test, test_rows)
    test_df = build_ranker_frame(test_sub, test_seen, stack)

    ranker = build_ranker(cfg, stack.tables)
    setup_mlflow(cfg)
    with mlflow.start_run(run_name=f"{cfg.mlflow.ranker_run_name}-{kind}"):
        mlflow.set_tag("ranker_type", kind)
        mlflow.log_params(flatten_dict({"ranker": raw_cfg["ranker"], "retrieval": raw_cfg["retrieval"],
                                        "rerank": raw_cfg["rerank"]}))
        logger.info("训练精排模型: %s", kind)
        curves = ranker.fit(train_df, train_sub["hist"], valid_df, valid_sub["hist"])

        for group, group_metrics in curves.items():             # 如 valid_ndcg_at_10、train_loss
            for name, values in group_metrics.items():
                for step, value in enumerate(values):
                    mlflow.log_metric(f"{group}_{name.replace('@', '_at_')}", value, step=step)

        scores = ranker.predict(test_df, test_sub["hist"])
        metrics = evaluate_variants(test_df, scores, stack.reranker, len(test_rows), rk.eval_ks)
        test_metrics = {f"test_{k}": v for k, v in metrics.items()}
        mlflow.log_metrics(test_metrics)
        logger.info("端到端测试指标: %s", {k: round(v, 4) for k, v in metrics.items()})

        path = ranker_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        ranker.save(path)
        mlflow.log_artifact(str(path), artifact_path="ranker")
        # 保存训练证据（指标 + 评估口径 + 文件指纹），注册模型时据此决定是否晋升 champion
        evidence = save_evidence(cfg, test_metrics, data_fingerprint(cfg, stack.vocab, len(test_rows)))
        mlflow.log_artifact(str(evidence), artifact_path="ranker")
        importance = ranker.feature_importance()
        if importance is not None:                                # 目前仅 LightGBM 提供
            importance_path = path.with_name(f"feature_importance_{kind}.csv")
            importance.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path), artifact_path="ranker")


if __name__ == "__main__":
    main()
