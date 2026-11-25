"""下载 Amazon Reviews 2023 的评论与商品元数据。

* 使用 certifi 的 CA 证书，避免 macOS 上 python.org 版 Python 报 CERTIFICATE_VERIFY_FAILED；
* 先写 .part 临时文件，下载完成再改名，中断后重跑不会读到半截文件；
* 目标文件已存在则跳过，所以也可以手动下载后放进 data/raw/。
"""
import logging
import os
import shutil
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import certifi

from twin_rank.paths import raw_meta_path, raw_review_path

logger = logging.getLogger(__name__)


def _ssl_context() -> ssl.SSLContext:
    """
    用 certifi 的 CA 证书，避免 macOS 上 python.org 版 Python 报
    CERTIFICATE_VERIFY_FAILED。公司代理做 HTTPS 拦截时，可设置
    SSL_CERT_FILE=/path/to/公司根证书.pem；实在不行再用 TWIN_RANK_INSECURE_SSL=1 临时跳过校验。
    """
    if os.environ.get("TWIN_RANK_INSECURE_SSL") == "1":
        logger.warning("已关闭 SSL 校验，仅建议临时使用")
        return ssl._create_unverified_context()
    return ssl.create_default_context(cafile=os.environ.get("SSL_CERT_FILE") or certifi.where())


def download_file(url: str, dest: Path) -> Path:
    if dest.exists():
        logger.info("已存在，跳过下载: %s", dest)
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    logger.info("下载 %s -> %s", url, dest)

    request = Request(url, headers={"User-Agent": "Twin-Rank/0.2"})
    with urlopen(request, context=_ssl_context(), timeout=60) as resp, open(tmp, "wb") as out:
        total = int(resp.headers.get("Content-Length") or 0)
        done, next_log = 0, 0
        while chunk := resp.read(1024 * 1024):
            out.write(chunk)
            done += len(chunk)
            if done >= next_log:
                logger.info("  已下载 %.0f MB%s", done / 1e6, f" / {total / 1e6:.0f} MB" if total else "")
                next_log += 100 * 1024 * 1024
    shutil.move(str(tmp), str(dest))
    return dest


def download_category(cfg) -> None:
    category = cfg.data.category
    download_file(cfg.data.review_url.format(category=category), raw_review_path(cfg))
    download_file(cfg.data.meta_url.format(category=category), raw_meta_path(cfg))
