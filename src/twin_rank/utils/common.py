"""通用工具：日志初始化、随机种子、训练设备选择（CUDA > Apple MPS > CPU）。"""
import logging
import random

import numpy as np
import torch

from twin_rank.utils.gpu import cuda_usable

logger = logging.getLogger(__name__)

CUDA_FIX_HINT = ("该 GPU 的算力不在当前 PyTorch 的支持列表里。要用这块 GPU，请重装带旧架构内核的版本，例如: "
                 "pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    """
    name: auto | cpu | cuda | mps。
    auto 依次尝试 CUDA、Apple MPS、CPU。选择 CUDA 前会确认当前 PyTorch 真的支持这块 GPU；
    不支持时（如新版 PyTorch + Tesla P100）退回 CPU 并给出修复提示，而不是训练到一半才报错。
    """
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        if cuda_usable():
            return torch.device("cuda")
        logger.warning("检测到 GPU 但当前 PyTorch 不支持它，改用 CPU。%s", CUDA_FIX_HINT)
        return torch.device("cpu")
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
