"""Twin-Rank：双塔向量召回 + LightGBM 精排 + MMR 重排的推荐系统。

本文件会在任何 twin_rank 子模块被导入前最先执行，用来设置平台相关的环境变量：

* macOS：torch / faiss / lightgbm 各自自带一份 libomp，同进程多线程会互相卡死，
  因此默认单线程（OMP_NUM_THREADS=1）并允许重复加载（KMP_DUPLICATE_LIB_OK）。
  KMP_DUPLICATE_LIB_OK 属于官方不推荐的绕过方式，仅用于本地开发；生产环境请用 Linux。
  同时开启 PyTorch 的 MPS 回退，遇到 Apple GPU 不支持的算子时自动改用 CPU。
* Linux / Windows：不需要任何处理。

以上变量都用 setdefault，你在命令行里显式设置的值优先，例如：
    OMP_NUM_THREADS=4 python -m twin_rank.pipelines.train_ranker
"""
import os
import sys

__version__ = "1.0.0"

if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
