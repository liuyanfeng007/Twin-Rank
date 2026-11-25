"""文件指纹：用来确认"评估指标"对应的就是"当前这份模型文件"。"""
import hashlib
from pathlib import Path


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(Path(path), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
