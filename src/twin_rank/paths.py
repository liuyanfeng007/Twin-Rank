"""所有中间文件与产物的路径集中在这里，避免各处拼字符串。

目录约定（都可在 config.yaml 的 data.* 里改）:
  data/raw        原始下载文件                data/processed  清洗后的交互/商品表
  data/features   编码器、物品特征、样本      data/artifacts  双塔权重、物品向量、Faiss 索引、精排模型
"""
from pathlib import Path


def raw_review_path(cfg) -> Path:
    return Path(cfg.data.raw_dir) / f"{cfg.data.category}.jsonl.gz"


def raw_meta_path(cfg) -> Path:
    return Path(cfg.data.raw_dir) / f"meta_{cfg.data.category}.jsonl.gz"


def interactions_path(cfg) -> Path:
    return Path(cfg.data.processed_dir) / "interactions.parquet"


def items_meta_path(cfg) -> Path:
    return Path(cfg.data.processed_dir) / "items_meta.parquet"


def encoded_interactions_path(cfg) -> Path:
    return Path(cfg.data.processed_dir) / "interactions_encoded.parquet"


def feature_dir(cfg) -> Path:
    return Path(cfg.data.feature_dir)


def artifact_dir(cfg) -> Path:
    return Path(cfg.data.artifact_dir)


def two_tower_path(cfg) -> Path:
    return artifact_dir(cfg) / "two_tower.pt"


def item_embeddings_path(cfg) -> Path:
    return artifact_dir(cfg) / "item_embeddings.npy"


def index_path(cfg) -> Path:
    return artifact_dir(cfg) / "items.faiss"


def ranker_type(cfg) -> str:
    """当前选择的精排模型类型；老配置没有该项时默认 lightgbm。"""
    return getattr(cfg.ranker, "model_type", "lightgbm")


def ranker_path(cfg) -> Path:
    """精排模型文件。LightGBM 存文本模型；深度模型按类型分文件，互不覆盖，方便对比。"""
    kind = ranker_type(cfg)
    name = cfg.ranker.model_name if kind == "lightgbm" else cfg.ranker.deep.model_name.format(model_type=kind)
    return artifact_dir(cfg) / name
