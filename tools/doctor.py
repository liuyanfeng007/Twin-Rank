"""环境自检：检查 Python 版本、关键依赖能否导入、平台相关的系统库，并给出针对性的修复建议。

用法: python tools/doctor.py        （start.sh / start.bat 会在训练前自动调用）
退出码: 0 = 必需项全部通过；1 = 有必需项失败。
"""
import importlib
import os
import platform
import shutil
import sys
import urllib.request

SYSTEM = platform.system()          # Darwin / Linux / Windows
FAILED = []


def report(ok: bool, name: str, detail: str = "", hint: str = "", required: bool = True):
    mark = "✔" if ok else ("✘" if required else "!")
    print(f"  {mark} {name} {detail}")
    if not ok:
        if hint:
            print(f"      → {hint}")
        if required:
            FAILED.append(name)


def check_python():
    v = sys.version_info
    ok = (3, 9) <= v[:2] <= (3, 12)
    report(ok, "Python", f"{v.major}.{v.minor}.{v.micro}",
           "推荐 3.9–3.12（过新的版本可能没有 torch / faiss 预编译包）", required=(v[:2] >= (3, 9)))


def check_import(module: str, hint: str = "", required: bool = True, show_version: bool = True):
    try:
        mod = importlib.import_module(module)
        report(True, module, getattr(mod, "__version__", "") if show_version else "")
        return mod
    except Exception as exc:                                    # noqa: BLE001  任何导入错误都要报告
        report(False, module, f"导入失败: {str(exc).splitlines()[0][:120]}", hint, required)
        return None


def lightgbm_hint() -> str:
    return {"Darwin": "macOS 缺少 OpenMP 运行库：brew install libomp",
            "Linux": "Linux 缺少 OpenMP 运行库：sudo apt-get install -y libgomp1（Debian/Ubuntu）",
            "Windows": "请安装 Microsoft Visual C++ Redistributable"}.get(SYSTEM, "")


def main():
    print(f"系统: {SYSTEM} {platform.machine()}  Python 可执行文件: {sys.executable}")
    check_python()

    import twin_rank  # noqa: F401  先导入：它会在 macOS 上设置 OpenMP 相关环境变量，必须早于 torch/faiss/lightgbm
    for name in ("numpy", "pandas", "pyarrow", "scipy", "yaml"):
        check_import(name, "pip install -e .", show_version=False)

    torch = check_import("torch", "pip install torch")          # 顺序很重要：torch → faiss → lightgbm
    if torch:
        from twin_rank.utils.common import CUDA_FIX_HINT, resolve_device
        from twin_rank.utils.gpu import cuda_usable
        if torch.cuda.is_available() and not cuda_usable():
            report(False, "GPU 兼容性", f"{torch.cuda.get_device_name(0)} 不被当前 PyTorch 支持，将自动使用 CPU",
                   CUDA_FIX_HINT, required=False)
        print(f"      训练设备(auto): {resolve_device('auto')}")
    check_import("faiss", "pip install faiss-cpu")
    check_import("lightgbm", lightgbm_hint())
    check_import("mlflow", "pip install mlflow")
    check_import("fastapi", "pip install fastapi uvicorn")
    check_import("reflex", "pip install -e '.[demo]'（仅 Web Demo 需要）", required=False)

    if SYSTEM in ("Linux", "Darwin") and not shutil.which("unzip"):
        report(False, "unzip", "", "Reflex 首次运行需要 unzip：sudo apt-get install -y unzip", required=False)

    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if uri.startswith("http"):
        try:
            urllib.request.urlopen(uri.rstrip("/") + "/health", timeout=3)
            report(True, "MLflow 服务", uri)
        except Exception as exc:                                # noqa: BLE001
            report(False, "MLflow 服务", f"{uri} 不可达: {exc}", "请先启动 MLflow 服务，或清空 MLFLOW_TRACKING_URI 使用本地存储",
                   required=False)
    else:
        print("  - MLflow: 未设置 MLFLOW_TRACKING_URI，将使用本地 ./mlruns 存储")

    print("\n结果:", "全部必需项通过 ✔" if not FAILED else f"以下必需项未通过: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
