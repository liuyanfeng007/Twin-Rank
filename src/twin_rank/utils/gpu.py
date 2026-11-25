"""判断当前安装的 PyTorch 是否真的能在这块 GPU 上运行。

背景：新版 PyTorch 的 CUDA 构建会逐步去掉老架构的内核（例如不再包含 Pascal 架构的 P100，算力 6.0）。
此时 torch.cuda.is_available() 仍为 True，但第一次计算会报 "no kernel image is available"。
所以选择 GPU 前要核对：这块卡的算力，是否在 PyTorch 编译的架构列表里。
"""


def _parse(arch: str):
    """'sm_75' -> ('sm', (7, 5))；'compute_120' -> ('compute', (12, 0))；'sm_90a' -> ('sm', (9, 0))。"""
    kind, _, num = arch.partition("_")
    digits = "".join(ch for ch in num if ch.isdigit())
    return kind, (int(digits[:-1]), int(digits[-1]))


def arch_supported(device_cc: tuple, arch_list: list) -> bool:
    """
    device_cc: GPU 算力，如 (6, 0)；arch_list: torch.cuda.get_arch_list()。
    * sm_XY（预编译内核）：主版本相同且 Y 不超过设备的次版本即可运行；
    * compute_XY（PTX 中间码）：只要不高于设备算力，就能在运行时编译。
    """
    for arch in arch_list:
        kind, cc = _parse(arch)
        if kind == "sm" and cc[0] == device_cc[0] and cc[1] <= device_cc[1]:
            return True
        if kind == "compute" and cc <= device_cc:
            return True
    return False


def cuda_usable() -> bool:
    """有 CUDA 且当前 PyTorch 构建支持这块 GPU 时返回 True。检查本身出错时不阻止使用（返回 True）。"""
    import torch
    if not torch.cuda.is_available():
        return False
    try:
        return arch_supported(torch.cuda.get_device_capability(0), torch.cuda.get_arch_list())
    except Exception:                                   # noqa: BLE001 检测失败不应让程序无法使用 GPU
        return True
