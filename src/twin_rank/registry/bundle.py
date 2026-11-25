"""把"完整推荐系统"打包成一个目录，作为 MLflow 模型的一个 artifact。

为什么要打包整套？线上预测不只需要精排模型：用户塔、物品向量、Faiss 索引、ID 编码器、
物品特征、商品目录、用户历史都缺一不可。把它们放在同一个版本里，才能保证
"注册表里的 v3" 加载出来就是可直接预测、彼此一致的整套系统。

目录布局（与项目的 data/ 目录一致，所以 RecommendService 无需改动即可从这里加载）:
    bundle/
      config.yaml
      data/features/   vocab.json, item_features.npy, user_encoder.json, item_encoder.json
      data/processed/  interactions_encoded.parquet (用户历史快照), items_meta.parquet
      data/artifacts/  two_tower.pt, item_embeddings.npy, items.faiss, <ranker>.txt
"""
import shutil
from pathlib import Path

import yaml

from twin_rank.config import _to_namespace
from twin_rank.paths import (artifact_dir, encoded_interactions_path, feature_dir, index_path,
                             item_embeddings_path, items_meta_path, ranker_path, two_tower_path)

FEATURE_FILES = ["vocab.json", "item_features.npy", "user_encoder.json", "item_encoder.json"]


def build_bundle(cfg, raw_cfg: dict, dest: Path) -> Path:
    """从当前 data/ 目录收集所有预测必需的文件到 dest，并写入配置副本。"""
    dest = Path(dest)
    for sub in ("features", "processed", "artifacts"):
        (dest / "data" / sub).mkdir(parents=True, exist_ok=True)

    for name in FEATURE_FILES:
        shutil.copy2(feature_dir(cfg) / name, dest / "data" / "features" / name)
    for src in (encoded_interactions_path(cfg), items_meta_path(cfg)):
        shutil.copy2(src, dest / "data" / "processed" / src.name)
    for src in (two_tower_path(cfg), item_embeddings_path(cfg), index_path(cfg), ranker_path(cfg)):
        shutil.copy2(src, dest / "data" / "artifacts" / src.name)

    bundled = dict(raw_cfg)
    bundled["serving"] = {**raw_cfg["serving"], "model_uri": None,
                          "cache": {**raw_cfg["serving"]["cache"], "enabled": False}}  # 注册模型不带缓存
    (dest / "config.yaml").write_text(yaml.safe_dump(bundled, allow_unicode=True, sort_keys=False),
                                      encoding="utf-8")
    return dest


def load_bundle_config(bundle: Path):
    """读取 bundle 里的配置，并把数据目录指向 bundle 内部（MLflow 会把 bundle 解压到任意本地路径）。"""
    bundle = Path(bundle)
    raw = yaml.safe_load((bundle / "config.yaml").read_text(encoding="utf-8"))
    raw["data"].update(
        feature_dir=str(bundle / "data" / "features"),
        processed_dir=str(bundle / "data" / "processed"),
        artifact_dir=str(bundle / "data" / "artifacts"),
    )
    return _to_namespace(raw)
