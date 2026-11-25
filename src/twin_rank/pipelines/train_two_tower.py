"""步骤 4：训练双塔召回模型，记录到 MLflow。

输出: data/artifacts/two_tower.pt、item_embeddings.npy；MLflow 里有参数、逐轮指标、测试集指标和模型。
"""

import json
import logging

import mlflow
import numpy as np
import pandas as pd
import torch

from twin_rank.models.two_tower import TwoTowerModel
from twin_rank.paths import artifact_dir, encoded_interactions_path, feature_dir
from twin_rank.pipelines._cli import init
from twin_rank.training.data import build_seen_matrix, load_arrays
from twin_rank.training.evaluate import evaluate
from twin_rank.training.loss import compute_log_q
from twin_rank.training.trainer import fit
from twin_rank.utils.common import resolve_device, set_seed
from twin_rank.utils.mlflow_utils import flatten_dict, log_pytorch_model, setup_mlflow

logger = logging.getLogger(__name__)

# 这些小文件随模型一起记录到 MLflow，后续做召回服务时可直接取用
SMALL_FEATURE_FILES = ["user_encoder.json", "item_encoder.json", "vocab.json", "item_features.npy"]


def main():
    cfg, raw_cfg = init("训练双塔召回模型")
    set_seed(cfg.train.seed)
    device = resolve_device(cfg.train.device)

    feat_dir, art_dir = feature_dir(cfg), artifact_dir(cfg)
    art_dir.mkdir(parents=True, exist_ok=True)

    vocab = json.loads((feat_dir / "vocab.json").read_text(encoding="utf-8"))
    item_feats = torch.from_numpy(np.load(feat_dir / "item_features.npy"))
    train, val, test = (load_arrays(feat_dir / f"{s}.npz") for s in ("train", "val", "test"))

    encoded = pd.read_parquet(encoded_interactions_path(cfg))
    val_seen = build_seen_matrix(encoded, ["train"], vocab)
    test_seen = build_seen_matrix(encoded, ["train", "val"], vocab)

    model = TwoTowerModel(item_feats, vocab, cfg.model).to(device)
    log_q = compute_log_q(train["item_idx"], vocab["n_items"])

    setup_mlflow(cfg)
    with mlflow.start_run(run_name=cfg.mlflow.run_name):
        mlflow.log_params(flatten_dict(raw_cfg))
        mlflow.log_params({"n_users": vocab["n_users"], "n_items": vocab["n_items"],
                           "n_train_samples": len(train["user_idx"])})

        best_state = fit(model, train, val, val_seen, log_q, cfg, device,
                         log_fn=lambda metrics, step: mlflow.log_metrics(metrics, step=step))

        model.load_state_dict(best_state)
        test_metrics = evaluate(model, test, test_seen, cfg.train.eval_ks, device)
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        logger.info("测试集: %s", {k: round(v, 4) for k, v in test_metrics.items()})

        # 导出：物品向量（下一步建 Faiss 索引用）、权重、特征文件
        model.cpu().eval()
        np.save(art_dir / "item_embeddings.npy", model.all_item_vectors().numpy())
        torch.save(model.state_dict(), art_dir / "two_tower.pt")
        for name in ("two_tower.pt", "item_embeddings.npy"):
            mlflow.log_artifact(str(art_dir / name), artifact_path="artifacts")
        for name in SMALL_FEATURE_FILES:
            mlflow.log_artifact(str(feat_dir / name), artifact_path="features")
        log_pytorch_model(model)


if __name__ == "__main__":
    main()
