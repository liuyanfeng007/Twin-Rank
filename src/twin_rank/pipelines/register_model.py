"""步骤 10：把整套系统(双塔+索引+精排+编码器+历史快照)注册到 MLflow Model Registry。

每次都会产生一个新版本；只有指标超过当前 champion 才会晋升为 champion（质量门禁），
否则只打 challenger 别名。--force 可强制晋升，--alias 可直接指定别名（跳过比较）。
"""
import argparse
import logging

from twin_rank.config import load_config
from twin_rank.registry.register import register_model
from twin_rank.utils.common import setup_logging

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="注册模型到 MLflow Model Registry（带质量门禁）")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--alias", help="直接指定别名（如 staging），不与 champion 比较")
    parser.add_argument("--force", action="store_true", help="无视比较结果，强制晋升为 champion")
    args = parser.parse_args()
    setup_logging()
    cfg, raw = load_config(args.config)
    result = register_model(cfg, raw, args.alias, args.force)
    logger.info("结果: %s", result)


if __name__ == "__main__":
    main()
