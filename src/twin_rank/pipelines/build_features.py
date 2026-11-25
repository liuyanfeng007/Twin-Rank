"""步骤 3：特征工程。

输入: data/processed/*.parquet
输出: data/features/ 下的 train/val/test.npz（用户、历史、目标）、item_features.npy、
      两个 ID 编码器 json、vocab.json；以及 data/processed/interactions_encoded.parquet
"""

import json
import logging

import numpy as np
import pandas as pd

from twin_rank.features.encoders import IdEncoder
from twin_rank.features.item_features import build_item_features
from twin_rank.features.samples import build_history, split_arrays
from twin_rank.paths import (encoded_interactions_path, feature_dir, interactions_path,
                          items_meta_path)
from twin_rank.pipelines._cli import init

logger = logging.getLogger(__name__)


def main():
    cfg, _ = init("特征工程：ID 编码、物品特征、序列样本")
    out_dir = feature_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(interactions_path(cfg))
    items_meta = pd.read_parquet(items_meta_path(cfg))

    # 1) ID 编码：只用训练集拟合，验证/测试里没见过的用户/物品会被编码为 0 并剔除
    train = df[df["split"] == "train"]
    user_enc = IdEncoder().fit(train["user_id"])
    item_enc = IdEncoder().fit(train["item_id"])
    df["user_idx"] = user_enc.transform(df["user_id"])
    df["item_idx"] = item_enc.transform(df["item_id"])
    df = df[(df["user_idx"] > 0) & (df["item_idx"] > 0)]
    df = df.sort_values(["user_idx", "timestamp"], kind="stable").reset_index(drop=True)

    # 2) 物品特征（热度仅用训练集）
    item_feats, vocab = build_item_features(
        items_meta, df[df["split"] == "train"], item_enc, cfg.features
    )
    vocab.update(n_users=len(user_enc), n_items=len(item_enc))

    # 3) 序列样本：历史包含该行之前的所有交互，因此测试样本的历史 = 训练 + 验证
    hist = build_history(df["user_idx"].to_numpy(), df["item_idx"].to_numpy(),
                         cfg.features.max_history_len)
    for split in ("train", "val", "test"):
        arrays = split_arrays(df, hist, split)
        np.savez_compressed(out_dir / f"{split}.npz", **arrays)
        logger.info("%s 样本数: %d", split, len(arrays["user_idx"]))

    # 4) 保存产物
    np.save(out_dir / "item_features.npy", item_feats)
    user_enc.save(out_dir / "user_encoder.json")
    item_enc.save(out_dir / "item_encoder.json")
    (out_dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
    df[["user_idx", "item_idx", "split"]].to_parquet(encoded_interactions_path(cfg), index=False)


if __name__ == "__main__":
    main()
