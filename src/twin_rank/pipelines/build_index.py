"""步骤 5：用物品向量构建 Faiss 索引。输出 data/artifacts/items.faiss。
"""

import logging

import numpy as np

from twin_rank.paths import index_path, item_embeddings_path
from twin_rank.pipelines._cli import init
from twin_rank.retrieval.index import VectorIndex

logger = logging.getLogger(__name__)


def main():
    cfg, _ = init("用物品塔向量构建 Faiss 索引")
    item_vecs = np.load(item_embeddings_path(cfg))
    r = cfg.retrieval
    index = VectorIndex.build(item_vecs, r.index_type, r.hnsw_m, r.hnsw_ef_search)
    index.save(index_path(cfg))
    logger.info("索引已保存: %s (%s, %d 个物品)", index_path(cfg), r.index_type, index.index.ntotal)


if __name__ == "__main__":
    main()
