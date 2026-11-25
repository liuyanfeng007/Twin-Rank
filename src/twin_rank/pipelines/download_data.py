"""步骤 1：下载 Amazon Reviews 数据。data.source=custom 时自动跳过。
"""

import logging

from twin_rank.data.download import download_category
from twin_rank.pipelines._cli import init

logger = logging.getLogger(__name__)


def main():
    cfg, _ = init("下载 Amazon Reviews 数据（data.source=custom 时自动跳过）")
    if cfg.data.source == "custom":
        logger.info("使用自定义数据 (%s)，跳过下载", cfg.data.custom.interactions_file)
        return
    download_category(cfg)


if __name__ == "__main__":
    main()
