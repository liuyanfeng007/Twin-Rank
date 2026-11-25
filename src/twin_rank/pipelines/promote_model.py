"""查看注册表里的各版本，或手动晋升 / 回滚。

    python -m twin_rank.pipelines.promote_model --list                 # 列出版本、别名、精排类型、指标
    python -m twin_rank.pipelines.promote_model --version 3            # 手动把版本 3 设为 champion（原 champion 留作 previous）
    python -m twin_rank.pipelines.promote_model --rollback             # 回滚：champion 指回 previous
"""
import argparse
import datetime
import logging

from mlflow.tracking import MlflowClient

from twin_rank.config import load_config
from twin_rank.registry import aliases
from twin_rank.utils.common import setup_logging
from twin_rank.utils.mlflow_utils import setup_mlflow

logger = logging.getLogger(__name__)


def list_versions(client, cfg) -> None:
    metric = cfg.registry.promote_metric
    versions = sorted(client.search_model_versions(f"name='{cfg.registry.model_name}'"),
                      key=lambda v: int(v.version))
    print(f"\n模型 {cfg.registry.model_name}（对比指标: {metric}）")
    print(f"{'版本':<6}{'别名':<24}{'精排':<10}{metric:<24}{'注册时间'}")
    for v in versions:
        evidence = aliases.read_evidence(v)
        value = f"{evidence[0].get(metric, float('nan')):.4f}" if evidence else "-（无指标）"
        created = datetime.datetime.fromtimestamp(v.creation_timestamp / 1000).strftime("%Y-%m-%d %H:%M")
        print(f"v{v.version:<5}{','.join(sorted(v.aliases)) or '-':<24}{(v.tags or {}).get('ranker_type', '-'):<10}"
              f"{value:<24}{created}")


def main():
    parser = argparse.ArgumentParser(description="查看版本 / 手动晋升 / 回滚")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--version", help="要晋升为 champion 的版本号")
    parser.add_argument("--rollback", action="store_true", help="champion 回到 previous")
    args = parser.parse_args()
    setup_logging()
    cfg, _ = load_config(args.config)
    setup_mlflow(cfg)
    client, reg = MlflowClient(), cfg.registry

    if args.rollback:
        previous = aliases.get_by_alias(client, reg.model_name, reg.previous_alias)
        if previous is None:
            raise SystemExit(f"没有 @{reg.previous_alias}，无法回滚")
        args.version = previous.version
    if args.version:
        aliases.promote(client, reg.model_name, args.version, reg.alias, reg.previous_alias)
        logger.info("已将版本 %s 设为 @%s", args.version, reg.alias)
    if args.list or not (args.version or args.rollback):
        list_versions(client, cfg)


if __name__ == "__main__":
    main()
