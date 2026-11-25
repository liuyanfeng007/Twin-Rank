"""GPU 架构兼容性判断（纯逻辑，不需要 torch）。"""
from twin_rank.utils.gpu import arch_supported

NEW_TORCH = ["sm_75", "sm_80", "sm_86", "sm_90", "sm_100", "sm_120", "compute_120"]   # 日志里的 cu132 构建
OLD_TORCH = ["sm_37", "sm_50", "sm_60", "sm_70", "sm_75", "sm_80", "sm_86", "compute_86"]


def test_p100_unsupported_by_new_torch_but_supported_by_old():
    assert not arch_supported((6, 0), NEW_TORCH)       # 你遇到的情况：Tesla P100 + torch 2.15 cu132
    assert arch_supported((6, 0), OLD_TORCH)


def test_common_gpus_with_new_torch():
    assert arch_supported((7, 5), NEW_TORCH)           # T4
    assert arch_supported((8, 0), NEW_TORCH)           # A100
    assert arch_supported((8, 9), NEW_TORCH)           # RTX 4090：sm_86 内核可向上兼容同一主版本
    assert arch_supported((9, 0), NEW_TORCH)           # H100
    assert arch_supported((12, 0), NEW_TORCH)          # Blackwell


def test_ptx_and_suffix_handling():
    assert arch_supported((9, 0), ["compute_80"])      # 低版本 PTX 可在新卡上编译
    assert not arch_supported((7, 0), ["compute_80"])  # 高版本 PTX 不能在旧卡上运行
    assert arch_supported((9, 0), ["sm_90a"])
