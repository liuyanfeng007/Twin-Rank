"""命令行预测：从 MLflow 注册表加载模型并推荐。

示例:
    python -m twin_rank.pipelines.predict --user-id AXXXX --top-k 10
    python -m twin_rank.pipelines.predict --user-id new_user --history B00AAA,B00BBB   # 指定历史
    python -m twin_rank.pipelines.predict --model-uri models:/twin-rank-recommender/3   # 指定版本
"""
import argparse
import json

import pandas as pd

from twin_rank.config import load_config
from twin_rank.paths import feature_dir
from twin_rank.registry.loader import default_model_uri, load_model
from twin_rank.utils.common import setup_logging
from twin_rank.utils.mlflow_utils import setup_mlflow


def main():
    parser = argparse.ArgumentParser(description="用 MLflow 注册表里的模型做推荐")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model-uri", help="默认 models:/<registry.model_name>@<registry.alias>")
    parser.add_argument("--user-id", help="原始用户 ID；省略则取库里的第一个用户")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--mode", default="full", choices=["recall", "rank", "full"])
    parser.add_argument("--history", help="逗号分隔的历史商品 ID（可选，提供后忽略库内历史）")
    parser.add_argument("--explain", action="store_true", help="同时输出推荐理由")
    args = parser.parse_args()

    setup_logging()
    cfg, _ = load_config(args.config)
    setup_mlflow(cfg)

    user_id = args.user_id
    if not user_id:                                       # 方便快速试用：取编码器里的第一个用户
        mapping = json.loads((feature_dir(cfg) / "user_encoder.json").read_text(encoding="utf-8"))
        user_id = next(iter(mapping))

    request = {"user_id": user_id, "top_k": args.top_k, "mode": args.mode, "explain": args.explain}
    if args.history:
        request["history"] = args.history
    model = load_model(args.model_uri or default_model_uri(cfg))
    result = model.predict(pd.DataFrame([request]))

    columns = ["final_rank", "item_id", "title", "store", "price", "source"]
    if args.mode != "recall":
        columns += ["recall_rank", "rank_score"]
    if args.explain:
        columns.append("reasons")
    print(f"\n用户: {user_id}  模式: {args.mode}")
    print(result[columns].to_string(index=False))


if __name__ == "__main__":
    main()
