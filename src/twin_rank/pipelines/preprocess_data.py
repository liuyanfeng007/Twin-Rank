"""步骤 2：清洗 + 时间切分。

输入: data/raw/*.jsonl.gz（或自定义数据文件）
输出: data/processed/interactions.parquet（含 split 列）、items_meta.parquet
"""

import logging

from twin_rank.data.cleaning import (META_COLUMNS, REVIEW_COLUMNS, build_interactions,
                                  clean_items_meta)
from twin_rank.data.custom import load_custom_meta, load_custom_reviews
from twin_rank.data.loaders import read_jsonl_gz
from twin_rank.data.split import assign_split
from twin_rank.paths import interactions_path, items_meta_path, raw_meta_path, raw_review_path
from twin_rank.pipelines._cli import init

logger = logging.getLogger(__name__)


def main():
    cfg, _ = init("数据预处理与时间切分")

    if cfg.data.source == "custom":
        reviews = load_custom_reviews(cfg.data.custom)
        meta = load_custom_meta(cfg.data.custom)
    else:
        reviews = read_jsonl_gz(raw_review_path(cfg), REVIEW_COLUMNS, cfg.preprocess.max_review_rows)
        meta = read_jsonl_gz(raw_meta_path(cfg), META_COLUMNS)
    interactions = assign_split(build_interactions(reviews, cfg.preprocess))
    logger.info("切分结果:\n%s", interactions["split"].value_counts())

    items_meta = clean_items_meta(meta, interactions["item_id"].unique())

    interactions_path(cfg).parent.mkdir(parents=True, exist_ok=True)
    interactions.to_parquet(interactions_path(cfg), index=False)
    items_meta.to_parquet(items_meta_path(cfg), index=False)


if __name__ == "__main__":
    main()
