"""把召回/重排需要的组件一次性装配好（Components），离线训练精排与在线服务共用。

装配内容: ID 编码器、用户塔、物品向量、Faiss 索引、热门召回、候选生成器、商品目录、重排器。
离线(train_ranker)和在线(RecommendService)走同一份装配代码，保证两边行为一致。
"""
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from twin_rank.features.encoders import IdEncoder
from twin_rank.paths import (encoded_interactions_path, feature_dir, index_path,
                             item_embeddings_path, items_meta_path)
from twin_rank.ranking.features import ItemTables
from twin_rank.rerank.reranker import Reranker
from twin_rank.retrieval.candidates import CandidateGenerator
from twin_rank.retrieval.index import VectorIndex
from twin_rank.retrieval.popular import PopularRecall
from twin_rank.retrieval.user_encoder import UserEncoder
from twin_rank.serving.catalog import ItemCatalog


@dataclass
class Components:
    vocab: dict
    encoded: pd.DataFrame          # 全量交互 (user_idx, item_idx, split)，按用户、时间排序
    user_enc: IdEncoder
    item_enc: IdEncoder
    user_encoder: UserEncoder
    item_vecs: np.ndarray
    tables: ItemTables
    popular: PopularRecall
    generator: CandidateGenerator
    catalog: ItemCatalog
    reranker: Reranker


def load_components(cfg) -> Components:
    feat_dir = feature_dir(cfg)
    vocab = json.loads((feat_dir / "vocab.json").read_text(encoding="utf-8"))
    item_feats = np.load(feat_dir / "item_features.npy")
    user_enc = IdEncoder.load(feat_dir / "user_encoder.json")
    item_enc = IdEncoder.load(feat_dir / "item_encoder.json")

    encoded = pd.read_parquet(encoded_interactions_path(cfg))
    train_items = encoded.loc[encoded["split"] == "train", "item_idx"].to_numpy()
    n_items = vocab["n_items"]

    item_vecs = np.load(item_embeddings_path(cfg))
    tables = ItemTables(item_feats, np.log1p(np.bincount(train_items, minlength=n_items)))
    popular = PopularRecall(train_items, n_items)
    generator = CandidateGenerator(VectorIndex.load(index_path(cfg)), item_vecs, popular,
                                   cfg.retrieval.n_candidates, cfg.retrieval.n_popular)
    catalog = ItemCatalog(pd.read_parquet(items_meta_path(cfg)), item_enc)
    reranker = Reranker(cfg.rerank, item_vecs, catalog.store_codes, item_enc)

    return Components(vocab, encoded, user_enc, item_enc,
                      UserEncoder.load(cfg, vocab, item_feats),
                      item_vecs, tables, popular, generator, catalog, reranker)
