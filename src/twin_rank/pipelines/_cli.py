"""所有流水线脚本共用：解析 --config、初始化日志、加载配置。
"""

import argparse

from twin_rank.config import load_config
from twin_rank.utils.common import setup_logging


def init(description: str):
    """所有流水线脚本共用：解析 --config、初始化日志、加载配置。"""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    setup_logging()
    return load_config(args.config)
